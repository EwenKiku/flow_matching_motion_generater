"""Checks for the robot configs and the flow-matching rollout."""
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch

from motion_flow.contact import StanceSlip
from motion_flow.data import Clip, load_clip, measured_twist, path_features, sample_training_batch, split_train_val
from motion_flow.model import Config, MotionFlow
from motion_flow.online import integrate_command
from motion_flow.robot import Robot, root_quat
from scripts.config import CONFIG_DIR, load_robot, make_generator


def _short_clip(source, dest, count):
    with np.load(source) as data:
        length = data['joint_pos'].shape[0]
        payload = {key: (data[key][:count] if data[key].ndim >= 1 and data[key].shape[0] == length else data[key])
                   for key in data.files}
    np.savez(dest, **payload)


class ConfigTests(unittest.TestCase):
    def test_default_robot_is_rp1_and_both_skeletons_load(self):
        spec = load_robot()
        self.assertEqual(spec.name, 'rp1')
        self.assertEqual(set(load_robot(name).name for name in ('rp1', 'pi_plus')), {'rp1', 'pi_plus'})
        with self.assertRaises(ValueError):
            load_robot('missing-robot')
        with self.assertRaises(ValueError):
            load_robot('rp1', CONFIG_DIR / 'pi_plus.yaml')
        for name in ('rp1', 'pi_plus'):
            robot_spec = load_robot(name)
            self.assertEqual(len(robot_spec.joints), {'rp1': 24, 'pi_plus': 22}[name])
            robot = Robot(robot_spec.xml, robot_spec.joints, robot_spec.bodies, robot_spec.anchor_body)
            self.assertEqual(robot.model.nq, len(robot_spec.joints) + 7)
            self.assertEqual(robot_spec.train.save_interval, 5000)
            self.assertGreaterEqual(robot_spec.train.pose_weight, 0)
            self.assertGreaterEqual(robot_spec.train.slip_weight, 0)

    def test_viewer_scene_loads_both_models(self):
        from scripts.keyboard_viewer import display_model
        for name in ('rp1', 'pi_plus'):
            spec = load_robot(name)
            model = display_model(spec.xml)
            self.assertGreater(model.ngeom, 0)
            self.assertLess(model.ngeom, 9000)
            self.assertEqual(model.nq, len(spec.joints) + 7)

    def test_rp1_export_matches_the_config(self):
        spec = load_robot('rp1')
        self.assertTrue(spec.motions.is_dir())
        self.assertTrue(spec.default_style.is_file())
        for path in (spec.default_style, spec.motions / 'alone_001__A116.npz'):
            clip = load_clip(path, spec.joints, spec.fps, spec.anchor_body)
            self.assertEqual(clip.joints, spec.joints)
            self.assertEqual(clip.bodies, spec.bodies)
            self.assertGreaterEqual(len(clip.pose), spec.model.history + spec.model.future)


def _mujoco_soles(robot, foot, xy, z, roll, pitch, yaw, joints):
    data = mujoco.MjData(robot.model)
    data.qpos[:2] = xy
    data.qpos[2] = z
    data.qpos[3:7] = root_quat(roll, pitch, yaw)
    data.qpos[robot.address] = joints
    mujoco.mj_kinematics(robot.model, data)
    local = np.asarray(foot.sole_point, dtype=np.float64)
    points = []
    for name in foot.bodies:
        body = robot.model.body(name).id
        points.append(data.xpos[body] + data.xmat[body].reshape(3, 3) @ local)
    return np.stack(points)


class AlgorithmTests(unittest.TestCase):
    def test_validation_split_keeps_one_training_clip(self):
        clips = list(range(10))
        train, val = split_train_val(clips, 0.2, np.random.default_rng(0))
        self.assertEqual(len(train) + len(val), 10)
        self.assertEqual(len(val), 2)
        self.assertEqual(set(train).isdisjoint(val), True)
        train, val = split_train_val(clips[:1], 0.2, np.random.default_rng(0))
        self.assertEqual(val, [])
        self.assertEqual(train, [clips[0]])
    def test_sole_positions_match_mujoco_and_slip_penalizes_support_motion(self):
        rng = np.random.default_rng(1)
        for name in ('rp1', 'pi_plus'):
            spec = load_robot(name)
            robot = Robot(spec.xml, spec.joints, spec.bodies, spec.anchor_body)
            stance = StanceSlip(spec.xml, spec.joints, spec.anchor_body, spec.foot, spec.fps)
            low = np.where(np.isfinite(robot.low), robot.low, -1.)
            high = np.where(np.isfinite(robot.high), robot.high, 1.)
            for _ in range(3):
                joints = rng.uniform(low, high).astype(np.float32)
                root = np.array([.2, -.1, .85, .05, -.08, .4], dtype=np.float32)
                pose = torch.tensor(np.r_[root[2:5], joints])
                got = stance.feet(pose, torch.tensor(root[:2]), torch.tensor(root[5])).detach().numpy()
                expected = _mujoco_soles(robot, spec.foot, root[:2], *root[2:], joints)
                np.testing.assert_allclose(got, expected, atol=1e-5)

        spec = load_robot('rp1')
        robot = Robot(spec.xml, spec.joints, spec.bodies, spec.anchor_body)
        stance = StanceSlip(spec.xml, spec.joints, spec.anchor_body, spec.foot, spec.fps)
        low = np.where(np.isfinite(robot.low), robot.low, -1.)
        high = np.where(np.isfinite(robot.high), robot.high, 1.)
        joints = np.clip(np.zeros(len(spec.joints), dtype=np.float32), low, high)
        pose = torch.tensor(np.r_[0.9, 0., 0., joints])
        xy = torch.zeros(2)
        sole_z = stance.feet(pose, xy, torch.zeros(())).detach()[:, 2]
        pose = pose.clone()
        pose[0] -= sole_z.min()
        pose = pose.repeat(2, 1)[None]
        anchor, future = pose[:, 0], pose[:, 1:]
        still = torch.zeros(1, 1, 4)
        moved = still.clone()
        moved[..., 0] = .02
        planted = stance(future, anchor, future, anchor, still, still)
        sliding = stance(future, anchor, future, anchor, moved, moved)
        self.assertLess(planted.item(), 1e-6)
        self.assertGreater(sliding.item(), planted.item() + 1e-3)
        future = future.detach().requires_grad_(True)
        sliding = stance(future, anchor, future.detach(), anchor, moved, moved)
        sliding.backward()
        self.assertGreater(future.grad[..., 3:].abs().sum().item(), 0)
    def test_double_support_supervises_both_legs(self):
        spec = load_robot('rp1')
        stance = StanceSlip(spec.xml, spec.joints, spec.anchor_body, spec.foot, spec.fps)
        pose = torch.zeros(1, 27)
        # Level both feet below the contact threshold.
        pose[:, 0] -= stance.feet(pose, torch.zeros(1, 2), torch.zeros(1))[..., 2].max()
        path = torch.zeros(1, 1, 4)
        for side in ('left', 'right'):
            pred = pose[:, None].clone()
            pred[..., 3 + spec.joints.index(side + '_knee_joint')] += .04
            self.assertGreater(stance(pred, pose, pose[:, None], pose, path, path).item(), .01)

    def test_stance_drift_resets_independently_at_touchdown(self):
        from motion_flow.contact import stance_drift
        ref = torch.zeros(1, 5, 2, 3)
        ref[..., 0] = torch.arange(5)[None, :, None] * .01
        ref[0, 2, 0, 2] = .1
        ref[0, :, 1, 2] = .1
        pred = ref.clone()
        pred[0, :, 0, 0] += torch.tensor([.2, .22, .6, .8, .84])
        pred.requires_grad_()
        loss = stance_drift(pred, ref, .025)
        self.assertAlmostEqual(loss.item(), (.02**2 + .04**2)/2, places=6)
        loss.backward()
        self.assertEqual(pred.grad[0, 2].abs().sum().item(), 0.)
        self.assertGreater(pred.grad[0, 4].abs().sum().item(), 0.)
        ref[..., 2] = .1
        self.assertEqual(stance_drift(pred, ref, .025).item(), 0.)

    def test_swing_tracks_reference_velocity(self):
        spec = load_robot()
        stance = StanceSlip(spec.xml, spec.joints, spec.anchor_body, spec.foot, spec.fps)
        anchor = torch.zeros(1, 3 + len(spec.joints))
        anchor[:, 0] = 2.  # Both feet clear the contact threshold.
        future = anchor[:, None].clone()
        future[..., 0] += .02
        path = torch.zeros(1, 1, 4)
        self.assertEqual(stance(future, anchor, future, anchor, path, path).item(), 0.)
        self.assertGreater(stance(anchor[:, None], anchor, future, anchor, path, path).item(), .9)

    def test_curved_command_is_independent_of_planning_horizon(self):
        origin = np.array([4., -3.])
        heading = np.pi - .005
        command = np.array([.5, -.2, 1.2])
        xy, yaw = integrate_command(origin, heading, command, 50, 30)
        current, angle = origin, heading
        for point, direction in zip(xy, yaw):
            stepped, stepped_yaw = integrate_command(current, angle, command, 50, 1)
            current, angle = stepped[0], stepped_yaw[0]
            np.testing.assert_allclose(current, point, atol=1e-12)
            self.assertAlmostEqual(angle, direction)
        local = path_features(xy, yaw, origin, heading)
        xy0, yaw0 = integrate_command(np.zeros(2), 0., command, 50, 30)
        np.testing.assert_allclose(local, path_features(xy0, yaw0, np.zeros(2), 0.), atol=1e-7)

    def test_root_twist_axes_and_yaw_wrap(self):
        quat = root_quat(.1, -.2, np.pi / 2)
        start = np.r_[1., 2., .4, quat]
        end = np.r_[.996, 1.994, .45, -quat]
        np.testing.assert_allclose(measured_twist(start, end, .02), [-.3, .2, 0.], atol=1e-12)
        heading = 3 * np.pi / 4
        start = np.r_[0., 0., .4, root_quat(.4, -.2, heading)]
        dx = np.cos(heading) - np.sin(heading)
        dy = np.sin(heading) + np.cos(heading)
        end = np.r_[dx, dy, .42, root_quat(-.2, .1, -heading)]
        np.testing.assert_allclose(measured_twist(start, end, np.pi / 2), [1., 0., 1.], atol=1e-12)

    def test_style_conditioning(self):
        torch.manual_seed(12)
        model = MotionFlow(Config(dof=2, width=16, layers=1)).eval()
        state = torch.randn(1, model.cfg.modes, 5)
        context = torch.randn(1, 12, model.cfg.width)
        time = torch.tensor([.4])
        with torch.inference_mode():
            small = model.velocity(state, time, context, style_bias=.5)
            large = model.velocity(state, time, context, style_bias=2.)
        self.assertGreater((small - large).abs().max().item(), 1e-4)
        style = torch.randn(1, 17, 5)
        padded = torch.cat((style, torch.randn(1, 23, 5)), dim=1)
        with torch.inference_mode():
            original = model.encode_style(style, torch.ones(1, 17, dtype=torch.bool))
            masked = model.encode_style(padded, torch.arange(40)[None] < 17)
        for actual, expected in zip(original, masked):
            torch.testing.assert_close(actual, expected)

    def test_rp1_batch_loss_is_finite(self):
        spec = load_robot('rp1')
        cfg = spec.model.build(len(spec.joints), spec.fps)
        rng = np.random.default_rng(0)
        pose = rng.normal(size=(96, cfg.dof + 3)).astype('f')
        clip = Clip(pose, np.zeros((96, 2), np.float32), np.zeros(96, np.float32), spec.joints, spec.bodies, spec.fps)
        *batch, speed = sample_training_batch([clip], 2, cfg, rng)
        net = MotionFlow(cfg)
        stance = StanceSlip(spec.xml, spec.joints, spec.anchor_body, spec.foot, spec.fps)
        loss = net.loss(*(torch.as_tensor(array) for array in batch), stance=stance,
                        pose_weight=spec.train.pose_weight, slip_weight=spec.train.slip_weight,
                        style_speed=torch.as_tensor(speed))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(net.last_slip))

    def test_history_links_match_mujoco_in_last_pelvis_heading(self):
        from motion_flow.data import rotate_xy
        spec = load_robot()
        cfg = spec.model.build(len(spec.joints), spec.fps)
        clip = load_clip(spec.default_style, spec.joints)
        sample = slice(25, 25 + cfg.history)
        nominal = clip.root_tracks[0][sample]
        track = path_features(nominal[:, :2], nominal[:, 2], np.array([1., -3.]), 1.2)
        net = MotionFlow(cfg)
        got = net.link_fk(torch.tensor(clip.features(sample, True))[None],
                          torch.tensor(track)[None], True)[0].numpy().reshape(cfg.history, -1, 9)
        robot = Robot(spec.xml, spec.joints, spec.bodies, spec.anchor_body)
        data = mujoco.MjData(robot.model)
        xy, yaw, pose = clip.xy[sample], clip.yaw[sample], clip.pose[sample]
        origin = np.r_[xy[-1], pose[-1, 0]]
        rotation = np.eye(3)
        rotation[:2, :2] = rotate_xy(np.eye(2), yaw[-1])
        ids = [robot.model.body(n).id for n in [spec.anchor_body] + [v[0] for v in cfg.history_links]]
        for i in range(cfg.history):
            data.qpos[:7] = np.r_[xy[i], pose[i, 0], root_quat(*pose[i, 1:3], yaw[i])]
            data.qpos[robot.address] = pose[i, 3:]
            mujoco.mj_kinematics(robot.model, data)
            xyz = (data.xpos[ids] - origin) @ rotation.T
            axes = rotation @ data.xmat[ids].reshape(-1, 3, 3)
            np.testing.assert_allclose(got[i, :, :3], xyz, atol=2e-6)
            np.testing.assert_allclose(got[i, :, 3:], axes[:, :, :2].transpose(0, 2, 1).reshape(-1, 6), atol=2e-6)

    def test_continuation_reanchors_path_and_keeps_gt_boundary(self):
        from scripts.train import continuation_batch
        spec = load_robot()
        cfg = spec.model.build(len(spec.joints), spec.fps)
        net = MotionFlow(cfg)
        stride = spec.infer.stride
        n = cfg.future + stride
        xy = np.column_stack((np.arange(n) * .01, np.arange(n)**2 * .001))
        yaw = np.linspace(2.9, 3.6, n)
        path = torch.tensor(path_features(xy, yaw, np.zeros(2), 0.))[None]
        future = torch.randn(1, n, cfg.channels)
        past = torch.cat((future[:, :cfg.history], path[:, :cfg.history]), dim=-1)
        valid = torch.ones(1, cfg.history, dtype=torch.bool)
        arrays, anchor = continuation_batch(net, [past, future[:, :cfg.history], valid, path, future],
                                            torch.ones(1), spec.infer)
        expected = path_features(xy[stride:], yaw[stride:], xy[stride-1], yaw[stride-1])
        np.testing.assert_allclose(arrays[3][0].numpy(), expected, atol=1e-6)
        torch.testing.assert_close(anchor, future[:, stride-1])
        torch.testing.assert_close(arrays[4], future[:, stride:])
        self.assertFalse(torch.equal(anchor, arrays[0][:, -1]))
        self.assertEqual(arrays[0].shape, past.shape)
        self.assertTrue(net.training)

    def test_mix_uses_one_arm_source_across_history_future_and_style(self):
        spec = load_robot()
        cfg = spec.model.build(len(spec.joints), spec.fps)
        clips = [Clip(np.full((80, cfg.dof + 3), v, dtype='f'), np.zeros((80, 2)),
                      np.zeros(80), spec.joints, spec.bodies, spec.fps) for v in (1, 2)]
        history, style, _, _, future, _ = sample_training_batch(
            clips, 8, cfg, np.random.default_rng(2), augment=True, arm_mix_prob=1.)
        arm = 3 + spec.joints.index('left_elbow_joint')
        for array in (history, future, style):
            np.testing.assert_array_equal(array[:, :, 0] + array[:, :, arm], 3)
            np.testing.assert_array_equal(array[:, 0, arm], history[:, 0, arm])
        np.testing.assert_array_equal(clips[0].pose, 1)
        np.testing.assert_array_equal(clips[1].pose, 2)


class RolloutTests(unittest.TestCase):
    def test_rp1_rollout_round_trip_and_robot_guard(self):
        spec = load_robot('rp1')
        cfg = spec.model.build(len(spec.joints), spec.fps)
        torch.manual_seed(0)
        net = MotionFlow(cfg)
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / 'rp1.pt'
            torch.save(dict(config=asdict(cfg), model=net.state_dict(), joints=spec.joints,
                            bodies=spec.bodies, robot=spec.name, step=0, seed=0,
                            pattern=['*.npz'], clips=1), checkpoint)
            style = folder / 'style.npz'
            _short_clip(spec.default_style, style, spec.model.history + 5)
            with self.assertRaises(SystemExit):
                make_generator(load_robot('pi_plus'), checkpoint, style)
            generator = make_generator(spec, checkpoint, style, device='cpu', seed=7)
            # Capture the sampled tensor before any planning/output processing.
            sample = generator.net.sample
            predictions = []
            def capture(*args):
                prediction = sample(*args).clone()
                self.assertEqual(prediction.shape[0], 1)
                prediction[..., -3:] = prediction.new_tensor([.03, -.01, .05])
                prediction[..., 3] = 5. + len(predictions)
                predictions.append(prediction[0].cpu().numpy().copy())
                return prediction
            generator.net.sample = capture
            frames = [generator.step((.25, 0., 0.)) for _ in range(generator.stride + 1)]
            np.testing.assert_allclose(frames[-1][7:], generator.plan[generator.cursor - 1, 3:3+cfg.dof])
            held = generator.command_xy.copy()
            generator.set_style(style)
            frames.append(generator.step((0., 0., 1.)))
            np.testing.assert_allclose(generator.command_xy, held)
            from motion_flow.data import rotate_xy
            np.testing.assert_allclose(frames[-1][:2], held + rotate_xy(np.array([.03, -.01]), generator.command_yaw))
            self.assertEqual(generator.cursor, 1)
            stacked = np.stack(frames)
            self.assertTrue(np.isfinite(stacked).all())
            np.testing.assert_allclose(np.linalg.norm(stacked[:, 3:7], axis=1), 1.)
            expected = np.concatenate((predictions[0][:generator.stride], predictions[1][:1],
                                       predictions[2][:1]))
            np.testing.assert_array_equal(stacked[:, 7:], expected[:, 3:3+cfg.dof])
            np.testing.assert_array_equal(stacked[:, 2], expected[:, 0])
            np.testing.assert_array_equal(generator.history[-1], expected[-1])
            saved = folder / 'trajectory.npz'
            generator.robot.save(saved, stacked, generator.fps)
            clip = load_clip(saved, spec.joints, spec.fps, spec.anchor_body)
            np.testing.assert_allclose(clip.xy, stacked[:, :2], atol=1e-5)
            np.testing.assert_allclose(clip.pose[:, 3:], stacked[:, 7:], atol=1e-5)
            with np.load(saved) as data:
                self.assertTrue(np.isfinite(data['body_ang_vel_w']).all())
                self.assertEqual(str(data['anchor_body'].item()), spec.anchor_body)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
