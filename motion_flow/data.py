from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Clip:
    pose: np.ndarray  # z, roll, pitch, then named hinge joints (radians)
    xy: np.ndarray
    yaw: np.ndarray
    joints: tuple[str, ...]
    bodies: tuple[str, ...]
    fps: float


def load_clip(path, joints=None, fps=None):
    with np.load(path, allow_pickle=False) as d:
        names = tuple(map(str, d['joint_names']))
        bodies = tuple(map(str, d['body_names']))
        root = bodies.index('base_link')
        hz = float(d['fps'].item())
        if hz <= 0 or not np.isfinite(hz) or (fps is not None and not np.isclose(fps, hz)):
            raise ValueError(f'{path}: unexpected fps {hz}, expected {fps}')
        joints = names if joints is None else tuple(joints)
        if len(set(names)) != len(names) or set(joints) != set(names):
            raise ValueError(f'{path}: joint_names do not match the skeleton')
        q = d['body_quat_w'][:, root].astype(np.float64)
        order = str(d['quaternion_order'].item())
        if order == 'xyzw':
            q = q[:, [3, 0, 1, 2]]
        elif order != 'wxyz':
            raise ValueError(f'{path}: unsupported quaternion_order {order}')
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError(f'{path}: degenerate root quaternion')
        w, x, y, z = (q / norm).T
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        pos = d['body_pos_w'][:, root]
        angles = d['joint_pos'][:, [names.index(name) for name in joints]]
        pose = np.column_stack((pos[:, 2], roll, pitch, angles)).astype('f')
        xy = pos[:, :2].astype('f')
        if not len(pose) or not all(np.isfinite(a).all() for a in (pose, xy, yaw)):
            raise ValueError(f'{path}: empty or nonfinite motion')
        return Clip(pose, xy, yaw, joints, bodies, hz)


def rotate_xy(xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = np.moveaxis(xy, -1, 0)
    return np.stack((c*x - s*y, s*x + c*y), axis=-1)


def path_features(xy, yaw, origin, heading):
    local = rotate_xy(xy - origin, -heading)
    angle = yaw - heading
    return np.column_stack((local, np.sin(angle), 1 - np.cos(angle))).astype('f')


def load_training_set(directory, patterns, max_clips, rng, history, future):
    paths = sorted({p for pattern in patterns for p in Path(directory).rglob(pattern)})
    if max_clips:
        rng.shuffle(paths)
        paths = paths[:max_clips]
    clips = []
    for index, path in enumerate(paths, start=1):
        first = clips[0] if clips else None
        clip = load_clip(path, first.joints if first else None, first.fps if first else None)
        if len(clip.pose) >= history + future:
            clips.append(clip)
        if index % 500 == 0 or index == len(paths):
            print(f'Loading motion files {index}/{len(paths)} | usable clips {len(clips)}', flush=True)
    if not clips:
        raise ValueError(f'No usable clips in {directory} matching {patterns}')
    return clips


def sample_batch(clips, batch_size, cfg, rng):
    history, future = cfg.history, cfg.future
    # Share the sampled style duration within a batch. Short clips use a padding mask.
    length = rng.integers(round(cfg.fps), round(5 * cfg.fps) + 1)
    past, styles, masks, paths, targets = [], [], [], [], []
    for idx in rng.integers(len(clips), size=batch_size):
        c = clips[idx]
        n = len(c.pose)
        start = rng.integers(n - history - future + 1)
        split = start + history
        end = split + future
        # Prefer disjoint windows, retaining short sequences when no disjoint crop fits.
        size = min(length, n)
        left = max(0, start - size + 1)
        right = max(0, n - end - size + 1)
        if left + right:
            offset = rng.integers(left + right)
            ref = offset if offset < left else end + offset - left
        else:
            ref = rng.integers(n - size + 1)
        style = c.pose[ref:ref+size]
        styles.append(np.pad(style, ((0, length-size), (0, 0)), mode='edge'))
        masks.append(np.arange(length) < size)
        past.append(c.pose[start:split])
        targets.append(c.pose[split:end])
        paths.append(path_features(c.xy[split:end], c.yaw[split:end], c.xy[split-1], c.yaw[split-1]))
    return tuple(np.stack(x) for x in (past, styles, masks, paths, targets))
