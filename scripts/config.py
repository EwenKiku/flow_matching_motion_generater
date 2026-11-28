"""Robot selection and the parameters that adapt the generator to a skeleton."""
from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from motion_flow.contact import FootConfig
from motion_flow.model import Config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs'

MODEL_KEYS = ('history', 'future', 'modes', 'width', 'layers', 'heads', 'style_seconds')
INFER_KEYS = ('steps', 'stride', 'style_bias', 'replan_delta')
TRAIN_KEYS = ('steps', 'batch_size', 'lr', 'seed', 'log_interval', 'threads', 'max_clips',
              'save_interval', 'val_fraction', 'val_batches',
              'flow_weight', 'slip_weight', 'drift_weight', 'arm_mix_prob', 'generated_history_prob', 'pattern')
FOOT_KEYS = ('bodies', 'sole_point', 'contact_height')
VIEWER_KEYS = ('title', 'distance', 'lookat_height', 'elevation', 'azimuth_offset', 'speed_step')
TOP_KEYS = ('name', 'xml', 'motions', 'checkpoint', 'default_style', 'anchor_body', 'fps',
            'joints', 'bodies', 'feet', 'viewer', 'model', 'infer', 'train')


@dataclass(frozen=True)
class ModelSpec:
    history: int
    future: int
    modes: int
    width: int
    layers: int
    heads: int
    style_seconds: float
    root_residual: bool
    history_links: list

    def build(self, dof, fps):
        return Config(dof=dof, fps=fps, history=self.history, future=self.future, modes=self.modes,
                      width=self.width, layers=self.layers, heads=self.heads, style_seconds=self.style_seconds, root_residual=self.root_residual, history_links=self.history_links)


@dataclass(frozen=True)
class InferSpec:
    steps: int
    stride: int
    style_bias: float
    replan_delta: float


@dataclass(frozen=True)
class TrainSpec:
    steps: int
    batch_size: int
    lr: float
    seed: int
    log_interval: int
    threads: int
    max_clips: int
    save_interval: int
    val_fraction: float
    val_batches: int
    flow_weight: float
    drift_weight: float
    slip_weight: float
    generated_history_prob: float
    arm_mix_prob: float
    pattern: tuple[str, ...]


@dataclass(frozen=True)
class ViewerSpec:
    title: str
    distance: float
    lookat_height: float
    elevation: float
    azimuth_offset: float
    speed_step: tuple[float, float, float]


@dataclass(frozen=True)
class RobotConfig:
    name: str
    path: Path
    xml: Path
    motions: Path
    checkpoint: Path
    default_style: Path
    anchor_body: str
    fps: float
    joints: tuple[str, ...]
    bodies: tuple[str, ...]
    foot: FootConfig
    viewer: ViewerSpec
    model: ModelSpec
    infer: InferSpec
    train: TrainSpec


def available_robots():
    return sorted(path.stem for path in CONFIG_DIR.glob('*.yaml'))


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a mapping')
    return value


def _keys(value, expected, label):
    missing = [key for key in expected if key not in value]
    extra = [key for key in value if key not in expected]
    if missing or extra:
        raise ValueError(f'{label} keys must be {list(expected)}; missing {missing}, unexpected {extra}')


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _names(value, label):
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f'{label} must be a non-empty list of names')
    return tuple(value)


def _number(value, label, kind=float):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a number')
    number = kind(value)
    if kind is int and int(value) != value:
        raise ValueError(f'{label} must be an integer')
    return number


def _positive(value, label, kind=float):
    number = _number(value, label, kind)
    if number <= 0:
        raise ValueError(f'{label} must be positive')
    return number


def load_robot(name=None, path=None):
    """Load configs/<name>.yaml, or an explicit file. The default robot is rp1."""
    if path is not None and name is not None:
        chosen = Path(path)
        spec = _load_file(chosen)
        if spec.name != name:
            raise ValueError(f'--robot {name} does not match {chosen} (name: {spec.name})')
        return spec
    if path is not None:
        return _load_file(Path(path))
    robot = name or 'rp1'
    file = CONFIG_DIR / f'{robot}.yaml'
    if not file.is_file():
        known = ', '.join(available_robots()) or '(none)'
        raise ValueError(f'Unknown robot {robot}. Configs in {CONFIG_DIR}: {known}')
    spec = _load_file(file)
    if spec.name != robot:
        raise ValueError(f'{file} declares name {spec.name}, expected {robot}')
    return spec


def _load_file(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Config file not found: {path}')
    with path.open(encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    raw = _mapping(raw, path.name)
    _keys(raw, TOP_KEYS, path.name)
    name = raw['name']
    if not isinstance(name, str) or not name:
        raise ValueError(f'{path}: name must be a string')
    joints = _names(raw['joints'], f'{path}: joints')
    if len(set(joints)) != len(joints):
        raise ValueError(f'{path}: joint names must be unique')
    bodies = _names(raw['bodies'], f'{path}: bodies')
    if raw['anchor_body'] not in bodies:
        raise ValueError(f'{path}: anchor_body must appear in bodies')
    fps = _positive(raw['fps'], f'{path}: fps')

    feet = _mapping(raw['feet'], f'{path}: feet')
    _keys({k: v for k, v in feet.items() if k != 'sole_half_length'}, FOOT_KEYS, f'{path}: feet')
    sole = feet['sole_point']
    if not isinstance(sole, list) or len(sole) != 3:
        raise ValueError(f'{path}: feet.sole_point must have three components')
    foot = FootConfig(
        bodies=_names(feet['bodies'], f'{path}: feet.bodies'),
        sole_point=tuple(_number(item, f'{path}: feet.sole_point') for item in sole),
        sole_half_length=_number(feet.get('sole_half_length', 0.), 'sole_half_length'),
        contact_height=_positive(feet['contact_height'], f'{path}: feet.contact_height'),
    )

    viewer_raw = _mapping(raw['viewer'], f'{path}: viewer')
    _keys(viewer_raw, VIEWER_KEYS, f'{path}: viewer')
    speed = viewer_raw['speed_step']
    if not isinstance(speed, list) or len(speed) != 3:
        raise ValueError(f'{path}: viewer.speed_step must have three components')
    speed = tuple(_positive(item, f'{path}: viewer.speed_step') for item in speed)
    if not isinstance(viewer_raw['title'], str) or not viewer_raw['title']:
        raise ValueError(f'{path}: viewer.title must be a string')
    viewer = ViewerSpec(
        title=viewer_raw['title'],
        distance=_positive(viewer_raw['distance'], f'{path}: viewer.distance'),
        lookat_height=_number(viewer_raw['lookat_height'], f'{path}: viewer.lookat_height'),
        elevation=_number(viewer_raw['elevation'], f'{path}: viewer.elevation'),
        azimuth_offset=_number(viewer_raw['azimuth_offset'], f'{path}: viewer.azimuth_offset'),
        speed_step=speed,
    )

    model_raw = _mapping(raw['model'], f'{path}: model')
    model_raw = dict(model_raw)
    root_residual = model_raw.pop('root_residual', False)
    if not isinstance(root_residual, bool):
        raise ValueError('model.root_residual must be boolean')
    link_history = model_raw.pop('link_history', False)
    if not isinstance(link_history, bool):
        raise ValueError('model.link_history must be boolean')
    from motion_flow.history import read_skeleton
    links = read_skeleton(_path(raw['xml']), raw['anchor_body'], joints) if link_history else []
    _keys(model_raw, MODEL_KEYS, f'{path}: model')
    model = ModelSpec(
        root_residual=root_residual, history_links=links,
        history=_positive(model_raw['history'], f'{path}: model.history', int),
        future=_positive(model_raw['future'], f'{path}: model.future', int),
        modes=_positive(model_raw['modes'], f'{path}: model.modes', int),
        width=_positive(model_raw['width'], f'{path}: model.width', int),
        layers=_positive(model_raw['layers'], f'{path}: model.layers', int),
        heads=_positive(model_raw['heads'], f'{path}: model.heads', int),
        style_seconds=_positive(model_raw['style_seconds'], f'{path}: model.style_seconds'),
    )
    if model.width % model.heads != 0:
        raise ValueError(f'{path}: model.width must be divisible by model.heads')
    if model.modes < 1 or model.future < 1:
        raise ValueError(f'{path}: model.future and model.modes must be positive')

    infer_raw = _mapping(raw['infer'], f'{path}: infer')
    infer_raw = dict(infer_raw)
    infer_raw.pop('blend', None)  # Older YAML files configured overlap blending.
    _keys(infer_raw, INFER_KEYS, f'{path}: infer')
    infer = InferSpec(
        steps=_positive(infer_raw['steps'], f'{path}: infer.steps', int),
        stride=_positive(infer_raw['stride'], f'{path}: infer.stride', int),
        style_bias=_number(infer_raw['style_bias'], f'{path}: infer.style_bias'),
        replan_delta=_number(infer_raw['replan_delta'], f'{path}: infer.replan_delta'),
    )
    if infer.replan_delta < 0 or not (1 <= infer.stride <= model.future):
        raise ValueError(f'{path}: invalid infer.stride or infer.replan_delta')

    train_raw = _mapping(raw['train'], f'{path}: train')
    train_raw = {'generated_history_prob': 0., 'drift_weight': 0., **train_raw}
    _keys(train_raw, TRAIN_KEYS, f'{path}: train')
    pattern = train_raw['pattern']
    if isinstance(pattern, str):
        pattern = [pattern]
    if not isinstance(pattern, list) or not pattern or not all(isinstance(item, str) and item for item in pattern):
        raise ValueError(f'{path}: train.pattern must be a list of glob patterns')
    train = TrainSpec(
        steps=_positive(train_raw['steps'], f'{path}: train.steps', int),
        batch_size=_positive(train_raw['batch_size'], f'{path}: train.batch_size', int),
        lr=_positive(train_raw['lr'], f'{path}: train.lr'),
        seed=_number(train_raw['seed'], f'{path}: train.seed', int),
        log_interval=_positive(train_raw['log_interval'], f'{path}: train.log_interval', int),
        threads=_positive(train_raw['threads'], f'{path}: train.threads', int),
        max_clips=_number(train_raw['max_clips'], f'{path}: train.max_clips', int),
        save_interval=_positive(train_raw['save_interval'], f'{path}: train.save_interval', int),
        val_fraction=_number(train_raw['val_fraction'], f'{path}: train.val_fraction'),
        val_batches=_positive(train_raw['val_batches'], f'{path}: train.val_batches', int),
        flow_weight=_number(train_raw['flow_weight'], f'{path}: train.flow_weight'),
        drift_weight=_number(train_raw['drift_weight'], f'{path}: train.drift_weight'),
        slip_weight=_number(train_raw['slip_weight'], f'{path}: train.slip_weight'),
        generated_history_prob=_number(train_raw['generated_history_prob'], 'train.generated_history_prob'),
        arm_mix_prob=_number(train_raw['arm_mix_prob'], 'train.arm_mix_prob'),
        pattern=tuple(pattern),
    )
    if not 0 <= train.generated_history_prob <= 1:
        raise ValueError("generated_history_prob must lie in [0, 1]")
    if not 0 <= train.arm_mix_prob <= 1:
        raise ValueError(f'{path}: train.arm_mix_prob must lie in [0, 1]')
    loss_weights = (train.flow_weight, train.slip_weight, train.drift_weight)
    if min(loss_weights) < 0 or not all(math.isfinite(weight) for weight in loss_weights):
        raise ValueError(f'{path}: loss weights must be nonnegative')
    if train.max_clips < 0:
        raise ValueError(f'{path}: train.max_clips must be nonnegative')
    if not 0 <= train.val_fraction < 1:
        raise ValueError(f'{path}: train.val_fraction must be in [0, 1)')

    xml = _path(raw['xml'])
    if not xml.is_file():
        raise ValueError(f'{path}: MJCF not found: {xml}')
    return RobotConfig(
        name=name, path=path.resolve(), xml=xml, motions=_path(raw['motions']),
        checkpoint=_path(raw['checkpoint']), default_style=_path(raw['default_style']),
        anchor_body=str(raw['anchor_body']), fps=fps, joints=joints, bodies=bodies,
        foot=foot, viewer=viewer, model=model, infer=infer, train=train,
    )


def add_robot_arguments(parser):
    parser.add_argument('--robot', help='configs/<robot>.yaml，默认 rp1')
    parser.add_argument('--config', type=Path, help='直接指定 YAML，写了此项就用文件里的 name')


def resolve_robot(robot, config):
    try:
        if config is not None:
            return load_robot(robot, config)
        return load_robot(robot)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def make_generator(spec, checkpoint, style, **kwargs):
    """Construct a generator and reject a checkpoint trained for a different robot."""
    import torch

    from motion_flow.online import TrajectoryGenerator

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f'找不到权重 {checkpoint}。先训练，或用 --checkpoint 指定。')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    trained = state.get('robot')
    if trained is not None and trained != spec.name:
        raise SystemExit(f'权重是给 {trained} 训练的，当前配置是 {spec.name}')
    if tuple(state['joints']) != spec.joints:
        raise SystemExit(f'权重的关节顺序与 {spec.path.name} 不一致')
    if tuple(state['bodies']) != spec.bodies:
        raise SystemExit(f'权重的刚体名单与 {spec.path.name} 不一致')
    infer = spec.infer
    options = dict(xml=spec.xml, anchor_body=spec.anchor_body,
                   steps=infer.steps, stride=infer.stride,
                   style_bias=infer.style_bias, replan_delta=infer.replan_delta)
    options.update(kwargs)
    return TrajectoryGenerator(checkpoint, style, **options)
