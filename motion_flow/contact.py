"""Bounded kinematic foot stabilization for the Pi Plus flat-ground reference."""
import mujoco
import numpy as np


class FootStabilizer:
    def __init__(self, robot, fps):
        self.robot = robot
        self.data = mujoco.MjData(robot.model)
        self.dt = 1 / fps
        self.feet = [robot.model.body(name).id for name in ('l_ankle_roll_link', 'r_ankle_roll_link')]
        self.columns = np.array([robot.model.joint(n).dofadr.item() for n in robot.joints])
        # Midsole point in the ankle-roll frame, from Pi Plus toe/heel collision geometry.
        self.local_point = np.array([-.025, 0., -.05])
        self.jacp = np.empty((3, robot.model.nv))
        self.jacr = np.empty_like(self.jacp)
        self.previous = None
        self.offset = np.zeros((2, 2))

    def _forward(self, frame):
        self.data.qpos[:7] = frame[:7]
        self.data.qpos[self.robot.address] = frame[7:]
        mujoco.mj_kinematics(self.robot.model, self.data)
        mujoco.mj_comPos(self.robot.model, self.data)
        rotation = self.data.xmat[self.feet].reshape(-1, 3, 3)
        return self.data.xpos[self.feet] + rotation @ self.local_point

    def reset(self, frame):
        self.previous = self._forward(frame)
        self.offset.fill(0.)

    def apply(self, frame):
        points = self._forward(frame)
        contact = np.clip(1 - points[:, 2] / .025, 0., 1.)
        contact = contact**2 * (3 - 2*contact)
        # Integrate a stationary-foot target in stance; release it smoothly in swing.
        self.offset += contact[:, None] * (self.previous[:, :2] - points[:, :2])
        self.offset *= np.exp(-(1-contact[:, None]) * self.dt / .045)
        self.offset /= np.maximum(1., np.linalg.norm(self.offset, axis=1, keepdims=True) / .065)
        self.previous = points
        target = points.copy()
        target[:, :2] += self.offset
        target[:, 2] = np.maximum(target[:, 2], 0)
        output = frame.copy()
        for _ in range(2):
            delta = np.zeros(len(self.columns))
            for i, foot in enumerate(self.feet):
                mujoco.mj_jac(self.robot.model, self.data, self.jacp, self.jacr, points[i], foot)
                # Preserve foot orientation while placing its sole. Non-leg columns are zero.
                jac = np.vstack((self.jacp[:, self.columns], .06 * self.jacr[:, self.columns]))
                error = np.r_[target[i] - points[i], np.zeros(3)]
                delta += jac.T @ np.linalg.solve(jac @ jac.T + .025**2 * np.eye(6), error)
            output[7:] = np.clip(output[7:] + np.clip(delta, -.22, .22),
                                 self.robot.low, self.robot.high)
            points = self._forward(output)
        return output
