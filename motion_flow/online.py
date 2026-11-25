"""Stateful, receding-horizon motion generation. Each step consumes one control tick."""
from collections import deque

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .contact import FootStabilizer
from .data import load_clip, path_features, rotate_xy
from .model import Config, MotionFlow
from .robot import DEFAULT_XML, Robot, root_quat


def integrate_command(xy, yaw, command, fps, count):
    """Integrate a constant body-frame twist in SE(2), including its zero-turn limit."""
    dt = np.arange(1, count + 1) / fps
    angle = command[2] * dt
    displacement = dt[:, None] * np.sinc(angle[:, None] / (2*np.pi)) * command[:2]
    return xy + rotate_xy(displacement, yaw + angle / 2), yaw + angle


class TrajectoryGenerator:
    def __init__(self, checkpoint, style, *, initial_clip=None, device='cpu', steps=5,
                 stride=10, blend=3, style_bias=1., seed=None, xml=DEFAULT_XML, stabilize_feet=True):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.cfg = Config(**state['config'])
        if not (1 <= stride <= self.cfg.future and 0 <= blend <= self.cfg.future - stride):
            raise ValueError('Require 1 <= stride <= future and 0 <= blend <= future - stride')
        if steps < 1 or not np.isfinite(style_bias):
            raise ValueError('Expected positive steps and finite style_bias')
        self.net = MotionFlow(self.cfg)
        self.net.load_state_dict(state['model'])
        self.net.to(device).eval()
        self.device, self.steps, self.stride = device, steps, stride
        self.blend, self.style_bias = blend, style_bias
        self.rng = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        self.robot = Robot(xml, state['joints'], state['bodies'])
        self.fps = self.cfg.fps
        self.feet = FootStabilizer(self.robot, self.fps) if stabilize_feet else None
        clip = self.set_style(style)
        if initial_clip is not None:
            clip = load_clip(initial_clip, self.robot.joints, self.fps)
        n = min(len(clip.pose), self.cfg.history)
        start = np.column_stack((clip.xy[:n], clip.pose[:n, 0],
                                 root_quat(clip.pose[:n, 1], clip.pose[:n, 2], clip.yaw[:n]),
                                 clip.pose[:n, 3:]))
        self.reset(start)

    @torch.inference_mode()
    def set_style(self, path):
        """Cache a new reference without resetting the current root or pose history."""
        c = load_clip(path, self.robot.joints, self.fps)
        embeddings, postures, lengths = [], [], []
        for chunk in np.array_split(c.pose, max(1, int(np.ceil(len(c.pose)/(5*self.fps))))):
            pose = self.net.normalize(torch.as_tensor(chunk[None], device=self.device))
            valid = torch.ones(pose.shape[:2], dtype=torch.bool, device=self.device)
            embedding, posture = self.net.encode_style(pose, valid)
            embeddings.append(embedding)
            postures.append(posture)
            lengths.append(len(chunk))
        weight = pose.new_tensor(lengths) / sum(lengths)
        embedding = torch.stack(embeddings).mul(weight[:, None, None]).sum(0)
        posture = torch.stack(postures).mul(weight[:, None, None]).sum(0)
        self.style = torch.nn.functional.normalize(embedding, dim=-1), posture
        self.force_replan = True
        return c

    def reset(self, history):
        """Set oldest-to-newest world [xyz, wxyz, checkpoint-order joints] frames."""
        a = np.atleast_2d(np.asarray(history, dtype=np.float64))[-self.cfg.history:]
        if a.shape != (len(a), self.cfg.dof + 7) or not len(a) or not np.isfinite(a).all():
            raise ValueError('Invalid initial history')
        rpy = Rotation.from_quat(a[:, [4, 5, 6, 3]]).as_euler('xyz')
        poses = np.column_stack((a[:, 2], rpy[:, :2], a[:, 7:])).astype('f')
        poses = np.pad(poses, ((self.cfg.history-len(a), 0), (0, 0)), mode='edge')
        self.history = deque(poses, maxlen=self.cfg.history)
        self.xy = a[-1, :2].copy()
        self.yaw = rpy[-1, 2]
        self.plan = np.empty((0, poses.shape[1]))
        self.cursor = 0
        self.force_replan = True
        if self.feet is not None:
            self.feet.reset(a[-1])

    def _plan(self, xy, yaw, command):
        path = path_features(xy, yaw, self.xy, self.yaw)
        past = torch.as_tensor(np.asarray(self.history)[None], device=self.device)
        future = self.net.sample(past, torch.as_tensor(path[None], device=self.device),
                                 self.style, self.steps, self.style_bias, self.rng)[0].cpu().numpy()
        overlap = min(self.blend, len(self.plan)-self.cursor)
        if overlap:
            weight = np.arange(1, overlap+1)[:, None] / (overlap+1)
            previous = self.plan[self.cursor:self.cursor+overlap]
            future[:overlap] = (1-weight)*previous + weight*future[:overlap]
        future[:, 3:] = np.clip(future[:, 3:], self.robot.low, self.robot.high)
        self.plan, self.cursor = future, 0
        self.planned_command = command.copy()
        self.force_replan = False

    def step(self, command):
        """Consume (forward m/s, left m/s, yaw rad/s); emit one world qpos reference."""
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('command must be a finite (vx, vy, yaw_rate) triple')
        replan = (self.force_replan or self.cursor >= self.stride or
                  np.max(np.abs(command - self.planned_command)) > .15)
        xy, yaw = integrate_command(self.xy, self.yaw, command, self.fps,
                                    self.cfg.future if replan else 1)
        if replan:
            self._plan(xy, yaw, command)
        pose = self.plan[self.cursor]
        self.cursor += 1
        self.history.append(pose)
        self.xy, self.yaw = xy[0], (yaw[0] + np.pi) % (2*np.pi) - np.pi
        frame = np.r_[self.xy, pose[0], root_quat(pose[1], pose[2], self.yaw), pose[3:]]
        return self.feet.apply(frame) if self.feet is not None else frame
