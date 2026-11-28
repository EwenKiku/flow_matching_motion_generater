"""Train the conditional flow field on reactor_mjlab retarget NPZs."""
import argparse
from dataclasses import asdict, replace
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from motion_flow.contact import StanceSlip
from motion_flow.data import measured_twist, load_training_set, sample_training_batch, split_train_val
from motion_flow.model import MotionFlow
from motion_flow.robot import root_quat
from scripts.config import make_generator, add_robot_arguments, resolve_robot


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def batch_tensors(clips, batch_size, cfg, rng, device, **sampling):
    past, style, valid, path, future, speed = sample_training_batch(clips, batch_size, cfg, rng, **sampling)
    arrays = [torch.as_tensor(array, device=device) for array in (past, style, valid, path, future)]
    return arrays, torch.as_tensor(speed, device=device)


def calibrate_coefficients(net, clips, batch_size, cfg, rng, device, arm_mix_prob, batches=32):
    """Match each sine coefficient to unit noise using training windows only."""
    count = 0
    total = square = None
    was_training = net.training
    net.eval()
    try:
        for _ in range(batches):
            (past, style, valid, _, future), _ = batch_tensors(
                clips, batch_size, cfg, rng, device, augment=True, arm_mix_prob=arm_mix_prob)
            with torch.no_grad():
                coeff = net.residual_coefficients(past, style, valid, future)
            count += coeff.shape[0]
            total = coeff.sum(0) if total is None else total + coeff.sum(0)
            square = coeff.square().sum(0) if square is None else square + coeff.square().sum(0)
    finally:
        net.train(was_training)
    mean = total / count
    std = (square / count - mean.square()).clamp(min=0).sqrt().clamp(min=1e-3)
    net.coeff_mean.copy_(mean)
    net.coeff_std.copy_(std)
    return std


def continuation_batch(net, arrays, speed, infer):
    """Commit one sampled stride and express the next GT path in its own anchor frame."""
    past, style, valid, path, future = arrays
    stride = infer.stride
    training = net.training
    net.eval()
    try:
        with torch.no_grad():
            embedding = net.encode_style(net.normalize(style), valid, speed)
            forecast = net.sample(past, path[:, :net.cfg.future], embedding,
                                  infer.steps, infer.style_bias)
            committed = forecast[:, :stride]
            if net.cfg.history_links:
                committed = torch.cat((committed, path[:, :stride]), -1)
            history = torch.cat((past, committed), dim=1)[:, -net.cfg.history:]
    finally:
        net.train(training)
    origin = path[:, stride-1:stride]
    tail = path[:, stride:]
    dx, dy = (tail[..., :2] - origin[..., :2]).unbind(-1)
    heading = torch.atan2(origin[..., 2], 1 - origin[..., 3])
    turn = torch.atan2(tail[..., 2], 1 - tail[..., 3]) - heading
    next_path = torch.stack((dx * heading.cos() + dy * heading.sin(),
                             dy * heading.cos() - dx * heading.sin(),
                             turn.sin(), 1 - turn.cos()), dim=-1)
    return [history, style, valid, next_path, future[:, stride:]], future[:, stride-1]


def evaluate(net, clips, batch_size, batches, cfg, rng, stance, train, device):
    """Mean loss on held-out clips. History dropout is off."""
    was_training = net.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.startswith('cuda') else None
    net.eval()
    total = 0.
    try:
        for _ in range(batches):
            arrays, speed = batch_tensors(clips, batch_size, cfg, rng, device)
            with torch.no_grad():
                value = net.loss(*arrays, stance=stance, style_speed=speed, pose_weight=train.pose_weight, slip_weight=train.slip_weight, drift_weight=train.drift_weight)
            total += value.item()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        net.train(was_training)
    return total / batches


@torch.inference_mode()
def rollout_error(spec, checkpoint, clips, stance, seed, device):
    """World-foot RMSE on fixed held-out continuations crossing plan boundaries."""
    with torch.random.fork_rng(devices=[]):
        generator = make_generator(spec, checkpoint, clips[0], seed=seed, device=device)
    h = generator.cfg.history
    errors = []
    for clip in clips[:4]:
        generator.set_style_clip(clip)
        n = min(len(clip.pose), h + 3 * generator.stride)
        pose = clip.pose[:n]
        frames = np.column_stack((clip.xy[:n], pose[:, 0],
                                  root_quat(pose[:, 1], pose[:, 2], clip.yaw[:n]), pose[:, 3:]))
        nominal = clip.root_tracks[0][:n] if generator.cfg.root_residual else None
        generator.reset(frames[:h], None if nominal is None else nominal[:h])
        commands = frames
        if nominal is not None:
            zeros = np.zeros(n)
            commands = np.column_stack((nominal[:, :2], zeros, root_quat(zeros, zeros, nominal[:, 2])))
        actual_roots = []
        generated = []
        for i in range(h, n):
            cmd = measured_twist(commands[i-1], commands[i], 1 / generator.fps)
            frame = generator.step(cmd)
            sample = np.r_[generator.history[-1][:3], frame[7:]].astype(np.float32)
            generated.append(sample)
            actual_roots.append([*generator.xy, generator.yaw])
        xy = torch.as_tensor(clip.xy[h:n], device=device)
        yaw = torch.as_tensor(clip.yaw[h:n], device=device)
        target = stance.feet(torch.as_tensor(pose[h:n], device=device), xy, yaw, contacts=True)
        roots = torch.as_tensor(actual_roots, device=device)
        actual = stance.feet(torch.as_tensor(np.stack(generated), device=device), roots[:, :2], roots[:, 2], contacts=True)
        errors.append((actual - target).square().sum(-1).mean().item())
    return math.sqrt(np.mean(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_robot_arguments(parser)
    parser.add_argument('--data', type=Path, help='覆盖配置里的动作目录')
    parser.add_argument('--pattern', nargs='+', help='覆盖配置里的文件名通配')
    parser.add_argument('--max-clips', type=int, help='0 表示使用全部匹配片段')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--out', type=Path, help='覆盖配置里的权重路径')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    spec = resolve_robot(args.robot, args.config)
    train = spec.train
    steps = args.steps if args.steps is not None else train.steps
    batch_size = args.batch_size if args.batch_size is not None else train.batch_size
    seed = args.seed if args.seed is not None else train.seed
    lr = args.lr if args.lr is not None else train.lr
    log_interval = args.log_interval if args.log_interval is not None else train.log_interval
    threads = args.threads if args.threads is not None else train.threads
    max_clips = args.max_clips if args.max_clips is not None else train.max_clips
    pattern = tuple(args.pattern) if args.pattern is not None else train.pattern
    motions = args.data if args.data is not None else spec.motions
    out = args.out if args.out is not None else spec.checkpoint
    if log_interval < 1:
        parser.error('--log-interval 必须为正')
    if min(steps, batch_size, threads) < 1 or max_clips < 0:
        parser.error('steps、batch-size、threads 必须为正，max-clips 不能为负')

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    rng = np.random.default_rng(seed)
    cfg = spec.model.build(len(spec.joints), spec.fps)
    print(f'Robot {spec.name} | config {spec.path} | {len(spec.joints)} joints | {spec.fps:g} Hz', flush=True)
    print(f'Loading motion data from {motions} ...', flush=True)
    clips = load_training_set(motions, pattern, max_clips, rng, cfg.history, cfg.future,
                              spec.joints, spec.fps, spec.anchor_body, spec.bodies)
    train_clips, val_clips = split_train_val(clips, train.val_fraction, rng)
    net = MotionFlow(cfg)
    # Moments come from the training clips only, so the holdout does not set the scale.
    total = sum(len(c.pose) for c in train_clips)
    mean = sum(c.features(slice(None), cfg.root_residual).sum(0, dtype=np.float64) for c in train_clips) / total
    variance = sum(((c.features(slice(None), cfg.root_residual) - mean)**2).sum(0) for c in train_clips) / total
    net.mean.copy_(torch.from_numpy(mean))
    net.scale.copy_(torch.from_numpy(np.maximum(variance**.5, .025)))
    net.to(args.device)
    coeff_std = calibrate_coefficients(net, train_clips, batch_size, cfg, np.random.default_rng(seed + 3),
                                       args.device, train.arm_mix_prob)
    print(f'coefficient std {coeff_std.min().item():.3f}..{coeff_std.max().item():.3f}', flush=True)
    stance = StanceSlip(spec.xml, spec.joints, spec.anchor_body, spec.foot, spec.fps).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * .1)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'{len(train_clips)} train clips, {len(val_clips)} val clips, '
          f'{total} train frames, {cfg.fps:g} Hz, device={args.device}', flush=True)
    best_path = out.with_name('best.pt')
    best_loss = math.inf
    running = 0.
    last_log_step = 0
    extended_cfg = replace(cfg, future=cfg.future + spec.infer.stride)
    long_clips = [c for c in train_clips if len(c.pose) >= cfg.history + extended_cfg.future]
    for step in range(1, steps + 1):
        continuation = bool(long_clips) and train.generated_history_prob > 0 and rng.random() < train.generated_history_prob
        arrays, speed = batch_tensors(long_clips if continuation else train_clips, batch_size,
                                      extended_cfg if continuation else cfg, rng, args.device,
                                      augment=True, arm_mix_prob=train.arm_mix_prob)
        ref_anchor = None
        if continuation:
            arrays, ref_anchor = continuation_batch(net, arrays, speed, spec.infer)
        loss = net.loss(*arrays, ref_anchor=ref_anchor, stance=stance, style_speed=speed, pose_weight=train.pose_weight, slip_weight=train.slip_weight, drift_weight=train.drift_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        schedule.step()
        current_loss = loss.detach().item()
        running += current_loss
        if step == 1 or step % log_interval == 0 or step == steps:
            mean_loss = running / (step - last_log_step)
            print(f'step {step}/{steps} ({100 * step / steps:.1f}%) | '
                  f'loss {current_loss:.4f} | pose {net.last_pose.item():.4f} | '
                  f'foot {net.last_slip.item():.4f} | drift {net.last_drift.item():.4f} | '
                  f'mean_loss {mean_loss:.4f}', flush=True)
            last_log_step = step
            running = 0.
        if step % train.save_interval == 0 or step == steps:
            payload = dict(config=asdict(cfg), model=net.state_dict(), joints=spec.joints,
                           bodies=spec.bodies, robot=spec.name, step=step, seed=seed,
                           pose_weight=train.pose_weight, slip_weight=train.slip_weight,
                           drift_weight=train.drift_weight, arm_mix_prob=train.arm_mix_prob,
                           generated_history_prob=train.generated_history_prob,
                           pattern=list(pattern), clips=len(train_clips), val_clips=len(val_clips))
            save_checkpoint(out, payload)
            if val_clips:
                val_loss = evaluate(net, val_clips, batch_size, train.val_batches, cfg, np.random.default_rng(seed + 1),
                                    stance, train, args.device)
                rmse = rollout_error(spec, out, val_clips, stance, seed, args.device)
                improved = rmse < best_loss
                print(f'step {step} | val_loss {val_loss:.4f} | rollout_foot_rmse {rmse:.4f} m'
                      + (' | saved best.pt' if improved else ''), flush=True)
                if improved:
                    best_loss = rmse
                    payload['rollout_foot_rmse'] = rmse
                    payload['val_loss'] = val_loss
                    save_checkpoint(best_path, payload)
            else:
                print(f'step {step} | 验证片段不足，跳过 val', flush=True)


if __name__ == '__main__':
    main()
