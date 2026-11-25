import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from motion_flow.data import load_clip, path_features
from motion_flow.model import Config, MotionFlow
from motion_flow.online import TrajectoryGenerator, integrate_command


class MotionTests(unittest.TestCase):
    def test_stream_switch_and_npz_round_trip(self):
        root = Path(__file__).resolve().parents[1]
        style = root / 'examples/style_switch.npz'
        torch.set_num_threads(1)
        gen = TrajectoryGenerator(root / 'checkpoints/pi_plus_walk.pt', style, seed=7)
        frames = [gen.step((.25, 0., 0.)) for _ in range(12)]
        xy = gen.xy.copy()
        gen.set_style(style)
        frames.append(gen.step((0., 0., 1.)))
        np.testing.assert_allclose(frames[-1][:2], xy)
        self.assertEqual(gen.cursor, 1)
        frames = np.stack(frames)
        np.testing.assert_allclose(np.linalg.norm(frames[:, 3:7], axis=1), 1.)
        with TemporaryDirectory() as folder:
            file = Path(folder) / 'trajectory.npz'
            gen.robot.save(file, frames, gen.fps)
            clip = load_clip(file, gen.robot.joints, gen.fps)
            np.testing.assert_allclose(clip.xy, frames[:, :2], atol=1e-6)
            np.testing.assert_allclose(clip.pose[:, 3:], frames[:, 7:], atol=1e-6)
            with np.load(file) as data:
                self.assertTrue(np.isfinite(data['body_ang_vel_w']).all())

    def test_curved_command_is_independent_of_planning_horizon(self):
        origin = np.array([4., -3.])
        heading = np.pi - .005
        command = np.array([.5, -.2, 1.2])
        xy, yaw = integrate_command(origin, heading, command, 50, 30)
        current, angle = origin, heading
        for point, direction in zip(xy, yaw):
            p, a = integrate_command(current, angle, command, 50, 1)
            current, angle = p[0], a[0]
            np.testing.assert_allclose(current, point, atol=1e-12)
            self.assertAlmostEqual(angle, direction)
        local = path_features(xy, yaw, origin, heading)
        xy0, yaw0 = integrate_command(np.zeros(2), 0., command, 50, 30)
        np.testing.assert_allclose(local, path_features(xy0, yaw0, np.zeros(2), 0.), atol=1e-7)

    def test_style_bias_preserves_magnitude_in_inference(self):
        torch.manual_seed(12)
        model = MotionFlow(Config(dof=2, width=16, layers=1)).eval()
        x = torch.randn(1, model.cfg.modes, 5)
        context = torch.randn(1, 12, model.cfg.width)
        t = torch.tensor([.4])
        with torch.inference_mode():
            small = model.velocity(x, t, context, style_bias=.5)
            large = model.velocity(x, t, context, style_bias=2.)
        self.assertGreater((small - large).abs().max().item(), 1e-4)

    def test_style_padding_does_not_change_condition(self):
        model = MotionFlow(Config(dof=2, width=16, layers=1)).eval()
        style = torch.randn(1, 17, 5)
        padded = torch.cat((style, torch.randn(1, 23, 5)), dim=1)
        with torch.inference_mode():
            a = model.encode_style(style, torch.ones(1, 17, dtype=torch.bool))
            b = model.encode_style(padded, torch.arange(40)[None] < 17)
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
