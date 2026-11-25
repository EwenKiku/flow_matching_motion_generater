from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    dof: int
    fps: float = 50.
    history: int = 15
    future: int = 30
    modes: int = 8
    width: int = 96
    layers: int = 3
    heads: int = 4


def sincos(time, width):
    freq = torch.exp(torch.arange(0, width, 2, device=time.device) * (-math.log(10000) / width))
    phase = time[..., None] * freq
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class MotionEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if src_mask is None:
            return super().forward(src, src_mask, src_key_padding_mask, is_causal)
        # The fused TransformerEncoderLayer can cast additive biases to booleans.
        # Use the public attention module directly to preserve real-valued style weights.
        h = self.norm1(src)
        attention, _ = self.self_attn(h, h, h, attn_mask=src_mask, need_weights=False,
                                      key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        src = src + self.dropout1(attention)
        h = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        return src + self.dropout2(h)


def encoder(cfg, layers):
    layer = MotionEncoderLayer(cfg.width, cfg.heads, cfg.width * 3,
                                       dropout=0., activation='gelu', batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(cfg.width),
                                 enable_nested_tensor=False)


class MotionFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim, width = cfg.dof + 3, cfg.width
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        # Anchored sine modes give zero displacement at frame zero and a free endpoint.
        frame = torch.arange(1, cfg.future+1)[:, None]
        mode = torch.arange(cfg.modes)
        basis = torch.sin(math.pi * frame * (mode + .5) / (cfg.future + .5))
        basis = basis / (1 + mode) ** 1.5
        self.register_buffer('basis', basis)
        self.register_buffer('inverse', torch.linalg.pinv(basis))
        self.style_in = nn.Linear(dim * 2, width)
        self.register = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.style_encoder = encoder(cfg, 2)
        self.history_in = nn.Linear(dim * 2, width)
        self.path_in = nn.Linear(7, width)
        self.mode_in = nn.Linear(dim, width)
        self.time_in = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.generator = encoder(cfg, cfg.layers)
        self.readout = nn.Linear(width, dim)
        self.style_inject = nn.Linear(width, width)
        self.register_buffer('positions', sincos(torch.arange(max(cfg.history, cfg.future)), width))

    def normalize(self, pose):
        return (pose - self.mean) / self.scale

    @staticmethod
    def with_delta(pose):
        delta = torch.diff(pose, dim=1, prepend=pose[:, :1])
        return torch.cat((pose, 5 * delta), -1)

    def encode_style(self, style, valid):
        # Extract phase-independent posture and dynamics. Tokens are spaced by 4 frames.
        mean = (style * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        samples = style[:, ::4]
        tokens = self.style_in(self.with_delta(samples))
        tokens = tokens + sincos(torch.arange(tokens.size(1), device=style.device), self.cfg.width)
        tokens = torch.cat((self.register.expand(len(style), -1, -1), tokens), 1)
        mask = F.pad(~valid[:, ::4], (1, 0), value=False)
        embedding = F.normalize(self.style_encoder(tokens, src_key_padding_mask=mask)[:, 0], dim=-1)
        return embedding, mean

    def context(self, past, path, style):
        embedding, posture = style
        h = self.history_in(self.with_delta(past)) + self.positions[:past.size(1)]
        if self.training:
            h = h * (torch.rand(len(h), 1, 1, device=h.device) >= .15)
        knots = path[:, 2::3]
        previous = F.pad(knots[:, :-1], (0, 0, 1, 0))
        delta_xy = knots[..., :2] - previous[..., :2]
        angle = torch.atan2(knots[..., 2], 1-knots[..., 3])
        delta_angle = torch.diff(angle, dim=1, prepend=torch.zeros_like(angle[:, :1]))
        twist = torch.cat((delta_xy, torch.atan2(delta_angle.sin(), delta_angle.cos())[..., None]), -1)
        cmd = self.path_in(torch.cat((knots, twist * (self.cfg.fps / 3)), -1))
        cmd = cmd + self.positions[2:self.cfg.future:3]
        style_token = self.style_inject(embedding * math.sqrt(self.cfg.width))
        # Center the residual around a gradual move toward the reference's average posture.
        fraction = torch.arange(1, self.cfg.future+1, device=past.device) / self.cfg.future
        baseline = past[:, -1:] + fraction[None, :, None] * (posture[:, None] - past[:, -1:])
        return torch.cat((style_token[:, None], h, cmd), 1), baseline

    def velocity(self, x, t, context, style_bias=0.):
        modes = self.mode_in(x) + self.positions[:self.cfg.modes]
        time = self.time_in(sincos(t * 100, self.cfg.width))[:, None]
        tokens = torch.cat((context, time, modes), 1)
        bias = None
        if style_bias:
            bias = x.new_zeros(tokens.size(1), tokens.size(1))
            bias[:, 0] = style_bias
        encoded = self.generator(tokens, mask=bias)
        # Analytic velocity of a unit Gaussian interpolant plus a learned residual.
        skip = (2*t - 1) / (t.square() + (1-t).square())
        return skip[:, None, None] * x + self.readout(encoded[:, -self.cfg.modes:])

    def loss(self, past, style, valid, path, future):
        past, style, future = [self.normalize(a) for a in (past, style, future)]
        # A joystick gives smooth intent, while measured root paths include gait sway.
        # Mix measured paths with constant body twists joining the same SE(2) endpoints.
        fraction = torch.arange(1, self.cfg.future+1, device=path.device) / self.cfg.future
        turn = torch.atan2(path[:, -1:, 2], 1-path[:, -1:, 3])
        angle = turn * fraction
        half_diff = (angle - turn) / 2
        x, y = path[:, -1:, 0], path[:, -1:, 1]
        ratio = fraction * torch.sinc(angle/(2*math.pi)) / torch.sinc(turn/(2*math.pi))
        smooth = torch.stack((ratio*(x*half_diff.cos()-y*half_diff.sin()),
                              ratio*(x*half_diff.sin()+y*half_diff.cos()),
                              angle.sin(), 1-angle.cos()), dim=-1)
        path = torch.where(torch.rand(len(path), 1, 1, device=path.device) < .7, smooth, path)
        context, baseline = self.context(past, path, self.encode_style(style, valid))
        target = self.inverse @ (future - baseline)
        noise = torch.randn_like(target)
        t = torch.rand(len(past), device=past.device)
        x = torch.lerp(noise, target, t[:, None, None])
        error = (self.velocity(x, t, context) - (target - noise)).square()
        # Emphasize the low modes responsible for stride and gross pose.
        weights = torch.arange(1, self.cfg.modes+1, device=x.device).rsqrt()
        return (error * (weights / weights.mean())[None, :, None]).mean()

    @torch.inference_mode()
    def sample(self, past, path, style, steps, style_bias=0., rng=None):
        if steps < 1:
            raise ValueError('steps must be positive')
        context, baseline = self.context(self.normalize(past), path, style)
        x = torch.randn((len(past), self.cfg.modes, past.size(-1)), device=past.device,
                        generator=rng)
        for i in range(steps):
            t = x.new_full((len(x),), i / steps)
            x = x + self.velocity(x, t, context, style_bias) / steps
        return (baseline + self.basis @ x) * self.scale + self.mean
