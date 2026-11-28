# 风格条件流匹配轨迹生成

给定一条机器人参考动作和每帧速度指令 `(vx, vy, wz)`，生成器以 50 Hz 输出浮动基座位姿和全部铰链关节角。风格参考可以在运行中切换。同一套流匹配网络同时支持 RP1 和 Pi Plus，机器人差异放在 `configs/` 里，默认是 RP1。

动作数据来自 [reactor_mjlab](https://github.com/EwenKiku/reactor_mjlab) 导出的重定向轨迹。当前仓库里的 `data/rp1` 就是这批 RP1 结果。Pi Plus 的导出放到 `data/pi_plus` 后，用 `--robot pi_plus` 训练和生成。

## 目录

| 路径 | 内容 |
| --- | --- |
| `motion_flow/` | 流匹配网络、训练采样、滚动生成、支撑脚损失 |
| `configs/rp1.yaml`、`configs/pi_plus.yaml` | 模型、关节顺序、足端几何、训练和推理参数 |
| `scripts/train.py` | 训练 |
| `scripts/generate.py` | 按速度指令导出一段 NPZ |
| `scripts/keyboard_viewer.py` | 键盘实时对照 |
| `scripts/test.py` | 配置和算法检查 |
| `data/rp1` | reactor_mjlab 的 RP1 重定向 NPZ |
| `robots/` | 两台机器人的 MJCF |

换机器人时改配置，不改网络代码。YAML 里的路径相对仓库根目录。`joints` 是网络通道顺序，加载时按关节名重排 NPZ。`feet.sole_point` 是踝滚转坐标系里的足底点，训练时用它惩罚支撑脚滑动。RP1 的训练损失分别监督足跟和前掌。

## 环境

在仓库根目录运行。依赖写在 `requirements.txt`，配置加载需要其中的 PyYAML。

```bash
conda activate flow_matching
python -m pip install -r requirements.txt
```

本机驱动是 550.144.03（CUDA 12.4），显卡是 RTX 3090。`requirements.txt` 把 PyTorch 固定为 `2.14.1+cu126`，这个构建可以在当前驱动上调用 GPU。CUDA 13 的轮子会报驱动过旧。训练不写 `--device` 时，检测到 CUDA 就用 GPU。

## 训练

```bash
python scripts/train.py --robot rp1 --data data/rp1
python scripts/train.py --robot pi_plus --data data/pi_plus --device cuda
```

不写 `--robot` 时使用 `configs/rp1.yaml`：读取 `data/rp1`，权重写到 `checkpoints/rp1.pt`。步数、批大小、学习率和片段通配都在配置的 `train` 段，命令行同名参数会盖过它们。`--max-clips 0` 表示使用全部够长的片段。

```bash
python scripts/train.py --robot rp1 --pattern 'walk*.npz' --max-clips 3600 --steps 10000 --device cuda
```

权重里会记下机器人名、关节顺序和刚体名单。生成时如果和当前配置对不上，会直接拒绝加载。默认每 5000 步保存一次当前权重，并用 5% 的片段做验证；固定验证片段的闭环足部位置 RMSE 创新低时写入同目录的 `best.pt`。

## 生成

```bash
python scripts/generate.py --style data/rp1/arc_walk_left_loop_001__A029.npz \
  --command 0.30 0.00 0.35 --seconds 6 --out /tmp/rp1_turn.npz
```

速度依次是当前朝向下的前进速度（m/s）、向左速度（m/s）和绕世界竖直轴的角速度（rad/s）。把 `--command` 和 `--seconds` 换成 `--commands commands.csv` 可以回放变化的指令。CSV 无表头，每行三个分量，行数就是输出长度。

导出的就是流模型这一帧的姿态，没有足端后处理。ODE 步数、重规划间隔和风格偏置在配置的 `infer` 段。

输出 NPZ 与 reactor 导出使用同一套字段：`fps`、`joint_names`、`body_names`、`joint_pos`、`joint_vel`，以及世界系刚体位姿和速度。四元数是 `wxyz`。

## 键盘实时对照

在服务器本机的图形桌面上直接运行。从另一台电脑看时，用 `ssh -X` 登录后再运行同一条命令。转发过来的屏幕没有可用的 OpenGL，查看器会在服务器 GPU 上画好，再把画面送到你当前的屏幕。

```bash
ssh -X kiku@192.168.10.186
conda activate flow_matching
python scripts/keyboard_viewer.py --robot rp1 --style data/rp1/alone_R_002__A187.npz
```

在服务器自己的桌面上运行时，仍然打开原来的 OpenGL 窗口。`ssh -X` 进来时，画面改由 GPU 画好后送到你当前的屏幕，左右两侧都会标出正在使用的风格名，下方是最近 10 秒的指令速度和机器人根速度。

左侧循环播放风格参考，右侧用同一段参考和当前速度指令实时生成。启动时三个速度分量都是零。点按一次改一档，松开后保持。窗口失去焦点时指令归零。左右方向键在同一目录里切换风格，切换后左侧参考和右侧生成都显示新的名字。

| 按键 | 作用 |
| --- | --- |
| W / S | 前进速度增加 / 减少一档 |
| A / D | 向左速度增加 / 减少一档 |
| Q / E | 左转角速度增加 / 减少一档 |
| 空格 | 三个速度立即归零 |
| ← / → | 切换上一条 / 下一条风格 |
| Esc | 退出 |

每档的大小在配置的 `viewer.speed_step`，也可以用 `--speed-step 0.05 0.05 0.10` 覆盖。鼠标左键旋转、右键平移、滚轮缩放，两个视口同步。

底部曲线是前向速度、侧向速度和偏航角速度。橙色是指令，青色是相邻两帧根位姿反算的速度。转弯时按机体朝向换算平移速度，并处理偏航角跨越 ±π。新 RP1 权重会在积分路径上预测骨盆偏移，曲线可反映步态引起的瞬时速度差。这组图不评估足底滑移和动力学跟踪。

## 接入控制循环

```python
import torch
from scripts.config import load_robot, make_generator

torch.set_num_threads(1)
spec = load_robot('rp1')  # 或 'pi_plus'
generator = make_generator(
    spec,
    spec.checkpoint,
    'data/rp1/arc_walk_left_loop_001__A029.npz',
    device='cpu',
    seed=42,
)
qpos = generator.step((0.30, 0.00, 0.35))
root_xyz, root_wxyz, joint_pos = qpos[:3], qpos[3:7], qpos[7:]
generator.set_style('data/rp1/arc_walk_left_loop_001__A030.npz')
```

`step` 每次推进一个采样周期，调用方负责按 50 Hz 调度。`set_style` 不重置根轨迹，下一个周期按新风格重规划。关节顺序是配置里的 `joints`，也就是 `generator.robot.joints`。

根轨迹每帧都跟着新指令走。姿态默认每 `infer.stride` 帧重规划；任一指令分量相对上次规划变化超过 `infer.replan_delta`，或切换风格时提前重规划。流模型保持自己的滚动历史。播放时直接使用模型输出的姿态。

## 训练方法

实现对应 `docs/walk_like_this.pdf` 第 4 节和附录 A.1 的风格双采样、历史差分、朝向坐标系、风格注意力偏置和重叠滚动生成。扩散去噪换成条件直线流匹配。

1. 从同一条动作里取一段历史和紧接着的未来，再另取 1 到 `style_seconds` 秒的风格片段。风格编码器每 4 帧取样，用 register token 汇总姿态和差分，输出单位范数嵌入。更长的参考在推理时分段编码后按时长加权。
2. 网络预测根高度、横滚、俯仰和各铰链角。先减去从当前姿态走向风格平均姿态的基线，再把未来窗口投影到从零位移起步的低频正弦模态。
3. 令目标系数为 `a`，采样噪声 `e` 和流时间 `t`，在 `(1-t)e + ta` 上回归速度 `a-e`。条件包括风格、历史、未来平面路径和速度。
4. RP1 训练把实测根路径分成平滑指令路径与局部骨盆偏移。推理时在 SE(2) 上积分指令，再叠加网络预测的水平位移和偏航。
5. 训练用参考足底高度计算支撑权重，支撑时惩罚水平滑动，摆动时匹配参考三维速度。总目标由流匹配和足部运动两项组成。
6. RP1 训练以 `arm_mix_prob` 的概率组合其他片段的手臂动作；验证使用原始轨迹。

检查配置和一轮前向：

```bash
python scripts/test.py
```
