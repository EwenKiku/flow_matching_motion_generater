"""Rigid-link history in the last pelvis frame, with world vertical preserved."""
import mujoco
import torch
from torch import nn

from .contact import _euler_xyz, _quat_matrix, _rodrigues


def read_skeleton(xml, anchor, joints):
    model = mujoco.MjModel.from_xml_path(str(xml))
    root = model.body(anchor).id
    indices = {root: 0}
    links = []
    for body in range(root + 1, model.nbody):
        parent = int(model.body_parentid[body])
        if parent not in indices:
            continue
        hinges = []
        for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError('Link history expects a hinge or a fixed body below the anchor')
            hinges.append((joints.index(model.joint(j).name), model.jnt_axis[j].tolist(),
                           model.jnt_pos[j].tolist(), float(model.qpos0[model.jnt_qposadr[j]])))
        links.append((model.body(body).name, indices[parent], model.body_pos[body].tolist(),
                      model.body_quat[body].tolist(), hinges))
        indices[body] = len(links)
    return links


class LinkHistory(nn.Module):
    def __init__(self, links):
        super().__init__()
        self.links = links
        for i, (_, _, pos, quat, hinges) in enumerate(links):
            self.register_buffer(f'p{i}', torch.tensor(pos), persistent=False)
            self.register_buffer(f'r{i}', torch.from_numpy(_quat_matrix(quat)), persistent=False)
            for k, (_, axis, pivot, _) in enumerate(hinges):
                self.register_buffer(f'a{i}_{k}', torch.tensor(axis), persistent=False)
                self.register_buffer(f'j{i}_{k}', torch.tensor(pivot), persistent=False)

    def forward(self, pose, track, residual):
        yaw = torch.atan2(track[..., 2], 1 - track[..., 3])
        xy = track[..., :2]
        if residual:
            dx, dy, turn = pose[..., -3:].unbind(-1)
            xy = xy + torch.stack((yaw.cos()*dx - yaw.sin()*dy,
                                   yaw.sin()*dx + yaw.cos()*dy), -1)
            yaw = yaw + turn
        dx, dy = (xy - xy[:, -1:]).unbind(-1)
        heading = yaw[:, -1:]
        z = pose[..., 0]
        base = torch.stack((heading.cos()*dx + heading.sin()*dy,
                            heading.cos()*dy - heading.sin()*dx, z - z[:, -1:]), -1)
        positions = [base]
        rotations = [_euler_xyz(pose[..., 1], pose[..., 2], yaw - heading)]
        for i, (_, parent, _, _, hinges) in enumerate(self.links):
            pos = positions[parent] + rotations[parent] @ getattr(self, f'p{i}')
            rot = rotations[parent] @ getattr(self, f'r{i}')
            for k, (joint, _, _, ref) in enumerate(hinges):
                pivot = getattr(self, f'j{i}_{k}')
                turned = rot @ _rodrigues(getattr(self, f'a{i}_{k}'), pose[..., 3+joint] - ref)
                pos = pos + (rot - turned) @ pivot
                rot = turned
            positions.append(pos)
            rotations.append(rot)
        xyz = torch.stack(positions, -2)
        axes = torch.stack(rotations, -3)[..., :2].transpose(-1, -2).flatten(-2)
        return torch.cat((xyz, axes), -1).flatten(-2)
