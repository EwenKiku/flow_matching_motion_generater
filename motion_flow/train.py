"""Train the conditional flow field on retargeted robot motion NPZs."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_flow.data import load_training_set, sample_batch
from motion_flow.model import Config, MotionFlow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/pi_plus_motions')
    parser.add_argument('--pattern', nargs='+', default=['*.npz'])
    parser.add_argument('--max-clips', type=int, default=0, help='0 loads all matching clips')
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Print training loss every N steps (default: 100), plus the first and last steps')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--out', type=Path, default=Path('checkpoints/pi_plus.pt'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error('--log-interval must be positive')
    if min(args.steps, args.batch_size, args.threads) < 1 or args.max_clips < 0:
        parser.error('steps, batch-size and threads must be positive and max-clips nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    rng = np.random.default_rng(args.seed)
    cfg = Config(dof=22)
    print(f'Loading motion data from {args.data} ...', flush=True)
    clips = load_training_set(args.data, args.pattern, args.max_clips, rng, cfg.history, cfg.future)
    first = clips[0]
    cfg.dof, cfg.fps = len(first.joints), first.fps
    net = MotionFlow(cfg)
    # Accumulate moments clip by clip to bound peak RAM on the full dataset.
    total = sum(len(c.pose) for c in clips)
    mean = sum(c.pose.sum(0, dtype=np.float64) for c in clips) / total
    variance = sum(((c.pose - mean)**2).sum(0) for c in clips) / total
    net.mean.copy_(torch.from_numpy(mean))
    net.scale.copy_(torch.from_numpy(np.maximum(variance**.5, .025)))
    net.to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * .1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f'{len(clips)} clips, {total} frames, {cfg.fps:g} Hz, device={args.device}', flush=True)
    running = 0.
    last_log_step = 0
    for step in range(1, args.steps+1):
        batch = sample_batch(clips, args.batch_size, cfg, rng)
        loss = net.loss(*(torch.as_tensor(a, device=args.device) for a in batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        schedule.step()
        current_loss = loss.detach().item()
        running += current_loss
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            mean_loss = running / (step - last_log_step)
            print(f'step {step}/{args.steps} ({100 * step / args.steps:.1f}%) | '
                  f'loss {current_loss:.4f} | mean_loss {mean_loss:.4f}', flush=True)
            last_log_step = step
            running = 0.
        if step % 1000 == 0 or step == args.steps:
            temp = args.out.with_suffix('.tmp')
            torch.save(dict(config=asdict(cfg), model=net.state_dict(), joints=first.joints,
                            bodies=first.bodies, step=step, seed=args.seed, pattern=args.pattern,
                            clips=len(clips)), temp)
            temp.replace(args.out)


if __name__ == '__main__':
    main()
