"""Roll out a constant velocity command, or a CSV of per-frame commands."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.config import add_robot_arguments, make_generator, resolve_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_robot_arguments(parser)
    parser.add_argument('--checkpoint', type=Path, help='默认用配置里的 checkpoint')
    parser.add_argument('--style', type=Path, help='默认用配置里的 default_style')
    parser.add_argument('--initial-clip', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--command', type=float, nargs=3, default=[.3, 0., 0.], metavar=('VX', 'VY', 'WZ'))
    group.add_argument('--commands', type=Path, help='无表头 CSV，每行是一个控制周期的 vx,vy,wz')
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--steps', type=int, help='流 ODE 的 Euler 步数')
    parser.add_argument('--stride', type=int)
    parser.add_argument('--style-bias', type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    spec = resolve_robot(args.robot, args.config)
    torch.set_num_threads(args.threads)
    style = args.style if args.style is not None else spec.default_style
    checkpoint = args.checkpoint if args.checkpoint is not None else spec.checkpoint
    overrides = {}
    if args.steps is not None:
        overrides['steps'] = args.steps
    if args.stride is not None:
        overrides['stride'] = args.stride
    if args.style_bias is not None:
        overrides['style_bias'] = args.style_bias
    generator = make_generator(
        spec, checkpoint, style, initial_clip=args.initial_clip, device=args.device,
        seed=spec.train.seed if args.seed is None else args.seed, **overrides)
    if args.commands:
        commands = np.loadtxt(args.commands, delimiter=',', ndmin=2)
    else:
        commands = np.tile(args.command, (round(generator.fps * args.seconds), 1))
    if commands.ndim != 2 or commands.shape[1] != 3 or not len(commands):
        parser.error('至少需要一条三维速度指令')
    frames = [generator.step(command) for command in commands]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generator.robot.save(args.out, frames, generator.fps)


if __name__ == '__main__':
    main()
