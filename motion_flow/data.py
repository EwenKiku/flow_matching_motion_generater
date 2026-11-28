from dataclasses import dataclass
from functools import cached_property
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

    @cached_property
    def root_tracks(self):
        """Low-frequency command and measured offsets in its instantaneous heading frame."""
        from scipy.ndimage import gaussian_filter1d

        yaw = np.unwrap(self.yaw.astype(np.float64))
        track = np.column_stack((self.xy, yaw))
        sigma = .15 * self.fps
        # Odd reflection preserves a straight path at the ends of the recording.
        pad = max(1, int(np.ceil(4 * sigma)))
        padded = np.pad(track, ((pad, pad), (0, 0)), mode='reflect', reflect_type='odd')
        nominal = gaussian_filter1d(padded, sigma, axis=0)[pad:-pad]
        delta = rotate_xy(self.xy - nominal[:, :2], -nominal[:, 2])
        residual = np.column_stack((delta, yaw - nominal[:, 2]))
        return nominal.astype('f'), residual.astype('f')

    def features(self, window, root_residual):
        pose = self.pose[window]
        if root_residual:
            return np.concatenate((pose, self.root_tracks[1][window]), axis=-1)
        return pose


def load_clip(path, joints=None, fps=None, anchor='base_link'):
    with np.load(path, allow_pickle=False) as d:
        names = tuple(map(str, d['joint_names']))
        bodies = tuple(map(str, d['body_names']))
        file_anchor = str(d['anchor_body'].item()) if 'anchor_body' in d.files else anchor
        if file_anchor != anchor:
            raise ValueError(f'{path}: anchor_body is {file_anchor}, expected {anchor}')
        if file_anchor not in bodies:
            raise ValueError(f'{path}: anchor {file_anchor} is not in body_names')
        root = bodies.index(file_anchor)
        hz = float(d['fps'].reshape(-1)[0])
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


# Walking windows are replayed at 0.8× to 1.25× their recorded root speed, using a constant time scale.
SPEED_RATIO = (0.8, 1.25)
LOCOMOTION_SPEED = 0.15
LOCOMOTION_JOINT = 0.25


def root_speed(xy, fps):
    """Median planar root speed, in m/s."""
    if len(xy) < 2:
        return 0.
    step = np.linalg.norm(np.diff(np.asarray(xy, dtype=np.float64), axis=0), axis=1)
    return float(np.median(step) * fps)


def _leg_columns(joints):
    index = [i for i, name in enumerate(joints)
             if 'hip_pitch' in name or 'knee' in name or 'calf' in name]
    if not index:
        return slice(3, None)
    return np.asarray(index, dtype=np.int64) + 3


def _sample_at(values, index):
    last = len(values) - 1
    i0 = np.floor(index).astype(np.int64)
    i1 = np.minimum(i0 + 1, last)
    weight = index - i0
    if values.ndim == 1:
        return (1 - weight) * values[i0] + weight * values[i1]
    return (1 - weight)[:, None] * values[i0] + weight[:, None] * values[i1]


def path_features(xy, yaw, origin, heading):
    local = rotate_xy(xy - origin, -heading)
    angle = yaw - heading
    return np.column_stack((local, np.sin(angle), 1 - np.cos(angle))).astype('f')


def measured_twist(previous, current, dt):
    """Body-frame planar twist over the interval between two emitted root poses."""
    def heading(frame):
        w, x, y, z = frame[3:7]
        return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))

    yaw = heading(previous)
    turn = (heading(current) - yaw + np.pi) % (2*np.pi) - np.pi
    # In a turn, the world displacement is a chord centered on the mid-step heading.
    local = rotate_xy(current[:2] - previous[:2], -yaw - turn/2)
    velocity = local / (dt * np.sinc(turn / (2*np.pi)))
    return np.array([velocity[0], velocity[1], turn / dt])


def load_training_set(directory, patterns, max_clips, rng, history, future,
                      joints=None, fps=None, anchor='base_link', bodies=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Motion directory does not exist: {directory}')
    paths = sorted({p for pattern in patterns for p in directory.rglob(pattern)})
    if max_clips:
        rng.shuffle(paths)
        paths = paths[:max_clips]
    locked_joints = None if joints is None else tuple(joints)
    locked_bodies = None if bodies is None else tuple(bodies)
    clips = []
    for index, path in enumerate(paths, start=1):
        clip = load_clip(path, locked_joints, fps, anchor)
        if locked_joints is None:
            locked_joints = clip.joints
        if locked_bodies is None:
            locked_bodies = clip.bodies
        elif clip.bodies != locked_bodies:
            raise ValueError(f'{path}: body_names do not match the robot config')
        if len(clip.pose) >= history + future:
            clips.append(clip)
        if index % 500 == 0 or index == len(paths):
            print(f'Loading motion files {index}/{len(paths)} | usable clips {len(clips)}', flush=True)
    if not clips:
        raise ValueError(f'No usable clips in {directory} matching {patterns}')
    return clips


def split_train_val(clips, fraction, rng):
    """Hold out a clip-level validation set. Moments and training must use the remainder."""
    if fraction < 0 or fraction >= 1:
        raise ValueError('val fraction must be in [0, 1)')
    if fraction == 0 or len(clips) < 2:
        return list(clips), []
    order = rng.permutation(len(clips))
    count = min(len(clips) - 1, max(1, int(round(len(clips) * fraction))))
    chosen = set(order[:count].tolist())
    train = [clip for index, clip in enumerate(clips) if index not in chosen]
    validation = [clip for index, clip in enumerate(clips) if index in chosen]
    return train, validation


def _style_window(clip, length, covered_lo, covered_hi, rng):
    """Original-rate style crop, disjoint from the source interval when the clip allows it."""
    n = len(clip.pose)
    size = min(length, n)
    left = max(0, covered_lo - size + 1)
    right = max(0, n - covered_hi - size + 1)
    if left + right:
        offset = rng.integers(left + right)
        ref = offset if offset < left else covered_hi + offset - left
    else:
        ref = rng.integers(n - size + 1)
    return ref, size


def sample_training_batch(clips, batch_size, cfg, rng, *, augment=False, arm_mix_prob=0.):
    """Sample aligned root, pose and style targets; augment training windows only."""
    history, future = cfg.history, cfg.future
    count = history + future
    length = rng.integers(round(cfg.fps), round(cfg.style_seconds * cfg.fps) + 1)
    low, high = np.log(SPEED_RATIO)
    past, styles, masks, paths, targets, speeds = [], [], [], [], [], []
    for idx in rng.integers(len(clips), size=batch_size):
        clip = clips[idx]
        n = len(clip.pose)
        start = int(rng.integers(0, n - count + 1))
        window = slice(start, start + count)
        speed = root_speed(clip.xy[window], clip.fps)
        amplitude = float(np.ptp(clip.pose[window, _leg_columns(clip.joints)], axis=0).max())
        rate = 1.
        if augment and speed >= LOCOMOTION_SPEED and amplitude >= LOCOMOTION_JOINT and rng.random() < .5:
            rate = min(float(np.exp(rng.uniform(low, high))), (n - start - 1) / (count - 1))
        if cfg.root_residual:
            track, _ = clip.root_tracks
            source_xy, source_yaw = track[:, :2], track[:, 2]
        else:
            source_xy, source_yaw = clip.xy, clip.yaw
        if rate == 1.:
            pose = clip.features(window, cfg.root_residual)
            xy, yaw = source_xy[window], source_yaw[window]
            covered_hi = start + count
        else:
            index = start + np.arange(count) * rate
            covered_hi = min(n, int(np.ceil(index[-1])) + 1)
            index = np.clip(index - start, 0, covered_hi - start - 1)
            source = slice(start, covered_hi)
            pose = _sample_at(clip.features(source, cfg.root_residual), index).astype('f')
            xy = _sample_at(source_xy[source], index).astype('f')
            yaw = _sample_at(np.unwrap(source_yaw[source]), index)
        ref, size = _style_window(clip, length, start, covered_hi, rng)
        style = clip.features(slice(ref, ref + size), cfg.root_residual)
        if augment and len(clips) > 1 and rng.random() < arm_mix_prob:
            donor = clips[(idx + rng.integers(1, len(clips))) % len(clips)]
            arms = [3 + j for j, name in enumerate(clip.joints)
                    if any(part in name for part in ('shoulder', 'elbow', 'wrist', 'upper_arm'))]
            offset = rng.integers(len(donor.pose) - count + 1)
            pose = pose.copy()
            pose[:, arms] = donor.pose[offset:offset + count, arms]
            donor_ref, donor_size = _style_window(donor, size, offset, offset + count, rng)
            size = min(size, donor_size)
            style = style[:size].copy()
            style[:, arms] = donor.pose[donor_ref:donor_ref + size, arms]
        history_pose = pose[:history]
        if cfg.history_links:
            track = path_features(xy[:history], yaw[:history], xy[history-1], yaw[history-1])
            history_pose = np.column_stack((history_pose, track))
        past.append(history_pose)
        targets.append(pose[history:])
        paths.append(path_features(xy[history:], yaw[history:], xy[history - 1], yaw[history - 1]))
        styles.append(np.pad(style, ((0, length - size), (0, 0)), mode='edge'))
        masks.append(np.arange(length) < size)
        speeds.append(root_speed(clip.xy[ref:ref + size], clip.fps))
    return (*tuple(np.stack(x) for x in (past, styles, masks, paths, targets)),
            np.asarray(speeds, dtype=np.float32))
