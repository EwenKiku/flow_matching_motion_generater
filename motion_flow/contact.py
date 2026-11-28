"""Differentiable foot motion supervision on the configured skeleton."""
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FootConfig:
    """Sole geometry for the stance loss. Inference does not read this."""

    bodies: tuple[str, ...]
    sole_point: tuple[float, float, float]
    contact_height: float

    sole_half_length: float = 0.

    def __post_init__(self):
        if not np.isfinite(self.sole_half_length) or self.sole_half_length < 0:
            raise ValueError("Invalid sole half length")
        point = tuple(self.sole_point)
        object.__setattr__(self, 'sole_point', point)
        if len(self.bodies) < 1:
            raise ValueError('feet.bodies must name at least one body')
        if len(point) != 3 or not np.isfinite(point).all():
            raise ValueError('feet.sole_point must be three finite coordinates in the ankle frame')
        if self.contact_height <= 0 or not np.isfinite(self.contact_height):
            raise ValueError('feet.contact_height must be a positive finite height')


def _quat_matrix(quat):
    w, x, y, z = (float(v) for v in quat)
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ), dtype=np.float32)


def _rodrigues(axis, angle):
    x, y, z = axis
    zero = torch.zeros((), dtype=angle.dtype, device=angle.device)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    identity = torch.eye(3, dtype=angle.dtype, device=angle.device)
    angle = angle[..., None, None]
    return identity + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew)


def _euler_xyz(roll, pitch, yaw):
    """Intrinsic xyz rotation, matching scipy's 'xyz' and the exported root quaternion."""
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    row0 = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1)
    row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=-1)
    row2 = torch.stack((-sp, cp * sr, cp * cr), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _support(height, limit):
    weight = ((limit - height) / limit).clamp(0, 1)
    return weight * weight * (3 - 2 * weight)


def stance_drift(predicted, reference, height_limit):
    """Mean squared horizontal displacement error since each point's touchdown."""
    contact = reference[..., 2] < height_limit
    frames = torch.arange(contact.size(1), device=contact.device)[None, :, None]
    # Each new contact run starts at its first grounded frame (including frame zero).
    touchdown = contact & ~torch.cat((torch.zeros_like(contact[:, :1]), contact[:, :-1]), dim=1)
    start = torch.where(touchdown, frames, 0).cummax(dim=1).values
    error = predicted[..., :2] - reference[..., :2]
    origin = error.gather(1, start.unsqueeze(-1).expand_as(error))
    active = contact & (frames > start)
    squared = (error - origin).square().sum(-1)
    return (squared * active).sum() / active.sum().clamp_min(1)


class StanceSlip(nn.Module):
    """Penalize stance slip and match the reference swing velocity."""

    def __init__(self, xml, joints, anchor_body, foot, fps):
        super().__init__()
        if fps <= 0:
            raise ValueError('fps must be positive')
        model = mujoco.MjModel.from_xml_path(str(xml))
        self.fps = float(fps)
        self.contact_height = float(foot.contact_height)
        self.half_length = foot.sole_half_length
        self.pose_dim = len(joints) + 3
        self.nfeet = len(foot.bodies)
        sole = torch.tensor(foot.sole_point, dtype=torch.float32)
        for foot_index, name in enumerate(foot.bodies):
            steps = _chain(model, anchor_body, name, joints)
            self.register_buffer(f'sole_{foot_index}', sole.clone())
            setattr(self, f'nsteps_{foot_index}', len(steps))
            for step_index, (pos, rot, qids, axes) in enumerate(steps):
                prefix = f'f{foot_index}s{step_index}'
                self.register_buffer(f'{prefix}_pos', torch.tensor(pos))
                self.register_buffer(f'{prefix}_rot', torch.tensor(rot))
                self.register_buffer(f'{prefix}_qid', torch.tensor(qids))
                self.register_buffer(f'{prefix}_axis', torch.tensor(axes))

    def feet(self, pose, xy, yaw, contacts=False):
        """Sole positions in the same frame as xy/yaw. pose is z, roll, pitch, joints."""
        dtype = getattr(self, 'sole_0').dtype
        pose, xy, yaw = pose.to(dtype), xy.to(dtype), yaw.to(dtype)
        origin = torch.cat((xy, pose[..., :1]), dim=-1)
        rotation = _euler_xyz(pose[..., 1], pose[..., 2], yaw)
        joints = pose[..., 3:]
        points = []
        for foot_index in range(self.nfeet):
            pos, rot = origin, rotation
            for step_index in range(getattr(self, f'nsteps_{foot_index}')):
                pos, rot = self._body(pos, rot, foot_index, step_index, joints)
            center = pos + torch.matmul(rot, getattr(self, f'sole_{foot_index}'))
            if contacts and self.half_length:
                offset = rot[..., :, 0] * self.half_length
                points.extend((center - offset, center + offset))
            else:
                points.append(center)
        return torch.stack(points, dim=-2)

    def _body(self, parent_pos, parent_rot, foot_index, step_index, joints):
        prefix = f'f{foot_index}s{step_index}'
        pos = parent_pos + torch.matmul(parent_rot, getattr(self, f'{prefix}_pos'))
        rot = torch.matmul(parent_rot, getattr(self, f'{prefix}_rot'))
        qids = getattr(self, f'{prefix}_qid')
        axes = getattr(self, f'{prefix}_axis')
        for joint in range(qids.shape[0]):
            rot = torch.matmul(rot, _rodrigues(axes[joint], joints[..., qids[joint]]))
        return pos, rot

    def forward(self, pred, anchor, ref, ref_anchor, path, measured_path):
        velocity, _ = self.terms(pred, anchor, ref, ref_anchor, path, measured_path)
        return velocity

    def terms(self, pred, anchor, ref, ref_anchor, path, measured_path, *, drift=False):
        predicted = self._sequence(anchor, pred, path)
        with torch.no_grad():
            reference = self._sequence(ref_anchor, ref, measured_path)
            height = reference[..., 2]
            weight = _support(torch.maximum(height[:, :-1], height[:, 1:]), self.contact_height)
        velocity = torch.diff(predicted, dim=1) * self.fps
        target_velocity = torch.diff(reference, dim=1) * self.fps
        slide = velocity[..., :2].square().sum(-1)
        swing_error = (velocity - target_velocity).square().sum(-1)
        velocity_loss = (weight * slide + (1 - weight) * swing_error).mean()
        displacement = stance_drift(predicted, reference, self.contact_height) if drift else velocity_loss.new_zeros(())
        return velocity_loss, displacement

    def _sequence(self, anchor, future, path):
        batch = anchor.shape[:-1]
        zeros = anchor.new_zeros(batch)
        xy = path[..., :2]
        yaw = torch.atan2(path[..., 2], 1 - path[..., 3])
        anchor_xy = torch.stack((zeros, zeros), dim=-1)
        anchor_yaw = zeros
        if future.size(-1) == self.pose_dim + 3:
            dx, dy, turn = future[..., -3:].unbind(-1)
            xy = xy + torch.stack((yaw.cos()*dx - yaw.sin()*dy,
                                   yaw.sin()*dx + yaw.cos()*dy), dim=-1)
            yaw = yaw + turn
            anchor_xy, anchor_yaw = anchor[..., -3:-1], anchor[..., -1]
        held = self.feet(anchor, anchor_xy, anchor_yaw, contacts=True)
        moved = self.feet(future, xy, yaw, contacts=True)
        return torch.cat((held.unsqueeze(-3), moved), dim=-3)


def _chain(model, anchor_body, foot_body, joints):
    anchor = model.body(anchor_body).id
    body = model.body(foot_body).id
    steps = []
    seen = 0
    while body != anchor:
        parent = int(model.body_parentid[body])
        if parent < 0 or seen > model.nbody:
            raise ValueError(f'{foot_body} is not a descendant of {anchor_body}')
        seen += 1
        address = int(model.body_jntadr[body])
        count = int(model.body_jntnum[body]) if address >= 0 else 0
        qids, axes = [], []
        for joint_id in range(address, address + count):
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f'{foot_body} chain contains a non-hinge joint')
            name = model.joint(joint_id).name
            if name not in joints:
                raise ValueError(f'{foot_body} chain uses {name}, which is not a configured joint')
            axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
            qids.append(joints.index(name))
            axes.append(axis / np.linalg.norm(axis))
        steps.append((
            np.asarray(model.body_pos[body], dtype=np.float32),
            _quat_matrix(model.body_quat[body]),
            np.asarray(qids, dtype=np.int64),
            np.asarray(axes, dtype=np.float32).reshape(-1, 3),
        ))
        body = parent
    steps.reverse()
    return steps
