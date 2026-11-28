"""Stateful, receding-horizon motion generation. Each step consumes one control tick."""
from collections import deque

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .data import Clip, load_clip, path_features, root_speed, rotate_xy
from .model import Config, MotionFlow
from .robot import Robot, root_quat


def integrate_command(xy, yaw, command, fps, count):
    """Integrate a constant body-frame twist in SE(2), including its zero-turn limit."""
    dt = np.arange(1, count + 1) / fps
    angle = command[2] * dt
    displacement = dt[:, None] * np.sinc(angle[:, None] / (2*np.pi)) * command[:2]
    return xy + rotate_xy(displacement, yaw + angle / 2), yaw + angle


class TrajectoryGenerator:
    def __init__(self, checkpoint, style, *, xml, anchor_body='base_link', initial_clip=None, foot=None,
                 device='cpu', steps=5, stride=10, blend=0, style_bias=1., replan_delta=0.15, seed=None):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.cfg = Config(**state['config'])
        # Legacy blend and foot keywords are accepted by older callers.
        if not 1 <= stride <= self.cfg.future:
            raise ValueError('Require 1 <= stride <= future')
        if steps < 1 or not np.isfinite(style_bias) or replan_delta < 0 or not np.isfinite(replan_delta):
            raise ValueError('Expected positive steps, finite style_bias, and nonnegative replan_delta')
        self.net = MotionFlow(self.cfg)
        self.net.load_state_dict(state['model'])
        self.net.to(device).eval()
        self.device, self.steps, self.stride = device, steps, stride
        self.style_bias, self.replan_delta = style_bias, replan_delta
        self.rng = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        self.anchor_body = anchor_body
        self.robot = Robot(xml, state['joints'], state['bodies'], anchor_body)
        self.fps = self.cfg.fps
        clip = self.set_style(style)
        if initial_clip is not None:
            clip = load_clip(initial_clip, self.robot.joints, self.fps, self.anchor_body)
        n = min(len(clip.pose), self.cfg.history)
        start = np.column_stack((clip.xy[:n], clip.pose[:n, 0],
                                 root_quat(clip.pose[:n, 1], clip.pose[:n, 2], clip.yaw[:n]),
                                 clip.pose[:n, 3:]))
        nominal = clip.root_tracks[0][:n] if self.cfg.root_residual else None
        self.reset(start, nominal)

    @torch.inference_mode()
    def set_style(self, path):
        """Cache a new reference without resetting the current root or pose history."""
        c = path if isinstance(path, Clip) else load_clip(path, self.robot.joints, self.fps, self.anchor_body)
        self.set_style_clip(c)
        return c

    @torch.inference_mode()
    def set_style_clip(self, c):
        embeddings, postures, lengths = [], [], []
        window = max(1, int(round(self.cfg.style_seconds * self.fps)))
        splits = max(1, int(np.ceil(len(c.pose) / window)))
        for index in np.array_split(np.arange(len(c.pose)), splits):
            chunk = c.features(index, self.cfg.root_residual)
            pose = self.net.normalize(torch.as_tensor(chunk[None], device=self.device))
            valid = torch.ones(pose.shape[:2], dtype=torch.bool, device=self.device)
            speed = pose.new_tensor([root_speed(c.xy[index], c.fps)])
            embedding, posture = self.net.encode_style(pose, valid, speed)
            embeddings.append(embedding)
            postures.append(posture)
            lengths.append(len(chunk))
        weight = pose.new_tensor(lengths) / sum(lengths)
        embedding = torch.stack(embeddings).mul(weight[:, None, None]).sum(0)
        posture = torch.stack(postures).mul(weight[:, None, None]).sum(0)
        self.style = torch.nn.functional.normalize(embedding, dim=-1), posture
        self.force_replan = True

    def reset(self, history, nominal=None):
        """Set oldest-to-newest world [xyz, wxyz, checkpoint-order joints] frames."""
        a = np.atleast_2d(np.asarray(history, dtype=np.float64))[-self.cfg.history:]
        if a.shape != (len(a), self.cfg.dof + 7) or not len(a) or not np.isfinite(a).all():
            raise ValueError('Invalid initial history')
        rpy = Rotation.from_quat(a[:, [4, 5, 6, 3]]).as_euler('xyz')
        poses = np.column_stack((a[:, 2], rpy[:, :2], a[:, 7:])).astype('f')
        self.command_xy = a[-1, :2].copy()
        self.command_yaw = rpy[-1, 2]
        if self.cfg.root_residual:
            residual = np.zeros((len(a), 3), dtype=np.float32)
            if nominal is not None:
                nominal = np.asarray(nominal)[-len(a):]
                if nominal.shape != residual.shape or not np.isfinite(nominal).all():
                    raise ValueError('Expected one finite nominal x/y/yaw per history frame')
                residual[:, :2] = rotate_xy(a[:, :2] - nominal[:, :2], -nominal[:, 2])
                residual[:, 2] = (rpy[:, 2] - nominal[:, 2] + np.pi) % (2*np.pi) - np.pi
                self.command_xy = nominal[-1, :2].astype(np.float64)
                self.command_yaw = float(nominal[-1, 2])
            poses = np.column_stack((poses, residual))
        poses = np.pad(poses, ((self.cfg.history-len(a), 0), (0, 0)), mode='edge')
        track = nominal if nominal is not None else np.column_stack((a[:, :2], rpy[:, 2]))
        track = np.pad(track, ((self.cfg.history-len(a), 0), (0, 0)), mode="edge")
        self.track_history = deque(track, maxlen=self.cfg.history)
        self.history = deque(poses, maxlen=self.cfg.history)
        self.xy = a[-1, :2].copy()
        self.yaw = rpy[-1, 2]
        self.plan = np.empty((0, poses.shape[1]))
        self.cursor = 0
        self.force_replan = True

    def _plan(self, xy, yaw, command):
        path = path_features(xy, yaw, self.command_xy, self.command_yaw)
        history = np.asarray(self.history)
        if self.cfg.history_links:
            track = np.asarray(self.track_history)
            relative = path_features(track[:, :2], track[:, 2], self.command_xy, self.command_yaw)
            history = np.column_stack((history, relative))
        past = torch.as_tensor(history[None], device=self.device)
        path = torch.as_tensor(path[None], device=self.device)
        sample = self.net.sample(past, path, self.style, self.steps, self.style_bias, self.rng)
        self.plan = sample[0].cpu().numpy()
        self.cursor = 0
        self.planned_command = command.copy()
        self.force_replan = False

    def step(self, command):
        """Consume (forward m/s, left m/s, yaw rad/s); emit one world qpos reference."""
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('command must be a finite (vx, vy, yaw_rate) triple')
        replan = (self.force_replan or self.cursor >= self.stride or
                  np.max(np.abs(command - self.planned_command)) > self.replan_delta)
        xy, yaw = integrate_command(self.command_xy, self.command_yaw, command, self.fps,
                                    self.cfg.future if replan else 1)
        if replan:
            self._plan(xy, yaw, command)
        pose = self.plan[self.cursor]
        self.cursor += 1
        self.command_xy, self.command_yaw = xy[0], (yaw[0] + np.pi) % (2*np.pi) - np.pi
        self.xy, self.yaw = self.command_xy, self.command_yaw
        if self.cfg.root_residual:
            self.xy = self.command_xy + rotate_xy(pose[-3:-1], self.command_yaw)
            self.yaw = self.command_yaw + pose[-1]
        self.track_history.append(np.r_[self.command_xy, self.command_yaw])
        self.history.append(pose)
        return np.r_[self.xy, pose[0], root_quat(pose[1], pose[2], self.yaw), pose[3:3+self.cfg.dof]]
