from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_XML = Path(__file__).resolve().parents[1] / 'robots/pi_plus/xmls/pi_plus.xml'


def root_quat(roll, pitch, yaw):
    quat = Rotation.from_euler('xyz', np.stack((roll, pitch, yaw), axis=-1)).as_quat()
    return np.roll(quat, 1, axis=-1)


class Robot:
    def __init__(self, xml, joints, bodies):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.joints, self.bodies = tuple(joints), tuple(bodies)
        self.address = [self.model.joint(n).qposadr.item() for n in joints]
        if self.model.nq != len(joints) + 7 or set(self.address) != set(range(7, self.model.nq)):
            raise ValueError('Expected a floating base followed by named hinge joints')
        self.body_ids = [self.model.body(n).id for n in bodies]
        ranges = [self.model.joint(n).range if self.model.joint(n).limited.item()
                  else [-np.inf, np.inf] for n in joints]
        self.low, self.high = np.asarray(ranges).T

    def save(self, output, frames, fps):
        """Export the same retarget NPZ schema as the input, with world-frame FK and velocities."""
        frames = np.asarray(frames)
        if frames.ndim != 2 or frames.shape[1] != 7 + len(self.joints) or len(frames) == 0:
            raise ValueError('Expected a nonempty trajectory in checkpoint joint order')
        data = mujoco.MjData(self.model)
        pos = np.empty((len(frames), len(self.bodies), 3))
        quat = np.empty((len(frames), len(self.bodies), 4))
        for i, frame in enumerate(frames):
            data.qpos[:7] = frame[:7]
            data.qpos[self.address] = frame[7:]
            mujoco.mj_kinematics(self.model, data)
            pos[i] = data.xpos[self.body_ids]
            quat[i] = data.xquat[self.body_ids]
        ang = np.zeros_like(pos)
        if len(frames) > 1:
            xyzw = np.roll(quat, -1, axis=-1)
            r0 = Rotation.from_quat(xyzw[:-1].reshape(-1, 4))
            r1 = Rotation.from_quat(xyzw[1:].reshape(-1, 4))
            diff = (r1 * r0.inv()).as_rotvec().reshape(pos[1:].shape) * fps
            ang[0], ang[-1] = diff[0], diff[-1]
            ang[1:-1] = (diff[:-1] + diff[1:]) / 2

        def velocity(x):
            return np.gradient(x, axis=0) * fps if len(x) > 1 else np.zeros_like(x)

        np.savez_compressed(output, fps=np.array([fps], dtype='f'), joint_names=self.joints,
                            body_names=self.bodies, anchor_body='base_link', quaternion_order='wxyz',
                            joint_pos=frames[:, 7:].astype('f'), joint_vel=velocity(frames[:, 7:]).astype('f'),
                            body_pos_w=pos.astype('f'), body_quat_w=quat.astype('f'),
                            body_lin_vel_w=velocity(pos).astype('f'), body_ang_vel_w=ang.astype('f'))
