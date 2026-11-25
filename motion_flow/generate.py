"""Roll out a constant velocity command or a sequence of per-frame commands."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .online import TrajectoryGenerator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--style', required=True)
    p.add_argument('--initial-clip')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--command', type=float, nargs=3, default=[.3, 0., 0.], metavar=('VX','VY','WZ'))
    g.add_argument('--commands', type=Path, help='CSV, one vx,vy,wz row per frame, no header')
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--style-bias', type=float, default=1.)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    gen = TrajectoryGenerator(args.checkpoint, args.style, initial_clip=args.initial_clip,
                              steps=args.steps, stride=args.stride, device=args.device,
                              seed=args.seed, style_bias=args.style_bias)
    if args.commands:
        commands = np.loadtxt(args.commands, delimiter=',', ndmin=2)
    else:
        commands = np.tile(args.command, (round(gen.fps*args.seconds), 1))
    if commands.ndim != 2 or commands.shape[1] != 3 or not len(commands):
        p.error('expected at least one 3-component velocity command')
    frames = [gen.step(command) for command in commands]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gen.robot.save(args.out, frames, gen.fps)


if __name__ == '__main__':
    main()
