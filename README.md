# Pi Plus 风格条件流匹配轨迹生成器

给定一段 Pi Plus 风格动作和每帧的速度指令 `(vx, vy, wz)`，生成器以 50 Hz 输出浮动基座位姿和 22 个关节角。风格参考可以在运行中切换。仓库包含可直接推理的权重 `checkpoints/pi_plus_walk.pt`。

[9 秒对照视频](examples/style_switch.mp4) 左侧播放风格参考，右侧从同一条连续生成的轨迹取帧。在第 3 秒切换为背手走并左转，第 6 秒切换为蹲走并改变侧向和转向速度。[对应的 450 帧轨迹](examples/style_switch.npz) 可以直接读取。

## 键盘实时对照

在有图形桌面的机器上，从仓库根目录启动 MuJoCo 双视口窗口。

```bash
conda activate reactor
python motion_flow/keyboard_viewer.py
```

左侧循环播放默认的背手步行参考，右侧使用同一段参考和当前速度指令实时生成动作。启动时三个速度分量均为零。每点按一次调整一档，松开后保持该速度。两侧地面的蓝线和橙线显示各自最近的根轨迹，右侧底部显示当前指令。

窗口底部三张曲线图分别对比前向速度、侧向速度（m/s）和偏航角速度（rad/s）。橙色为该步指令，青色为从相邻两帧输出根位姿反算的速度。青色方点在两线重合时仍可见。横轴是轨迹时间，保留最近 10 秒，在每个控制步末记录同一时间戳的一对样本。转弯时按机体朝向换算平移速度，并处理偏航角跨越 ±π。当前生成器直接积分指令得到平面根轨迹，因此两线通常重合。这组图衡量运动学根轨迹的速度，足底滑移和动力学跟踪需要单独评估。

在任一视口按住鼠标左键拖动可旋转视角，右键拖动可平移，滚轮向上拉近、向下拉远。左右视口同步调整，松开鼠标后各自以调整后的视角继续跟随机器人。

| 按键 | 右侧机器人的速度指令 |
| --- | --- |
| W / S | 前进速度增加 / 减少 0.05 m/s |
| A / D | 向左速度增加 / 减少 0.05 m/s |
| Q / E | 左转角速度增加 / 减少 0.10 rad/s |
| 空格 | 所有速度立即归零，松开后保持零 |
| Esc | 退出 |

例如从零连续点十次 W，前进速度会保持在 0.50 m/s。再点一次 S 减到 0.45 m/s，点一次 Q 就会边前进边左转。三个速度分量都按点按次数向正负方向累加。窗口失去焦点时指令归零。用 `--style` 选择参考片段，用 `--speed-step` 设置每次点按的增量。

```bash
python -m motion_flow.keyboard_viewer \
  --style data/pi_plus_motions/crouch_ff_loop_360_003__A125.npz \
  --speed-step 0.05 0.03 0.10
```

## 生成一段动作

在仓库根目录运行。当前机器的 `reactor` Conda 环境包含所需依赖。新环境可用 `python -m pip install -r requirements.txt` 安装。

```bash
conda activate reactor
python -m motion_flow.generate \
  --checkpoint checkpoints/pi_plus_walk.pt \
  --style data/pi_plus_motions/walk_hands_on_back_loop_001__A029.npz \
  --command 0.30 0.00 0.35 --seconds 6 --out /tmp/styled_walk.npz
```

该例生成一段背手向左转弯的步行轨迹。速度指令依次为当前朝向下的前进速度（m/s）、向左速度（m/s）和绕世界竖直轴向左的角速度（rad/s）。如需回放变化的指令，将 `--command ... --seconds ...` 替换为 `--commands commands.csv`。CSV 无表头，每行是一个控制周期的三个分量，行数决定输出长度。

输出 NPZ 与 `data/pi_plus_motions` 使用相同字段，包括 `fps`、`joint_names`、`body_names`、`joint_pos`、`joint_vel`、世界坐标系下的刚体位置、姿态、线速度和角速度。导出通过 Pi Plus MJCF 计算各连杆的正运动学，四元数采用 `wxyz`。

## 接入实时控制循环

```python
import torch
from motion_flow.online import TrajectoryGenerator

torch.set_num_threads(1)  # 单机器人 CPU 推理
generator = TrajectoryGenerator(
    "checkpoints/pi_plus_walk.pt",
    "data/pi_plus_motions/walk_ff_loop_360_003__A049.npz",
    seed=42,
)

# 在 50 Hz 循环内，每个周期用最新的速度调用一次
qpos = generator.step((0.30, 0.00, 0.35))
root_xyz, root_wxyz, joint_pos = qpos[:3], qpos[3:7], qpos[7:]
joint_names = generator.robot.joints

# 切换风格时调用一次，下一个周期按新风格重规划
generator.set_style("data/pi_plus_motions/crouch_ff_loop_360_003__A125.npz")
qpos = generator.step((0.20, 0.05, 0.10))
```

`step` 每次将时间推进一个采样周期，调用方负责调度。初始化默认取风格片段的前 15 帧作为历史，也可以用 `initial_clip=` 单独指定起始动作。与仿真或现有控制器连接时，可用 `generator.reset(history)` 设置世界系起始历史。每行按 `[xyz, wxyz, joint_pos]` 排列，关节顺序见 `generator.robot.joints`，历史从旧到新，短历史在开头复制首帧。低层跟踪器可以消费这些运动学参考，接触力、平衡和执行器力矩需要由低层控制器处理。

根轨迹在每一帧响应新指令。姿态默认每 10 帧重规划，任一指令分量相对上次规划变化超过 0.15 或切换风格时提前重规划。前 3 帧与旧窗口剩余部分混合。`steps` 控制流 ODE 的 Euler 步数，默认 5。`style_bias` 是施加到风格 token 上的注意力加数，默认 1。缓存风格和加载模型在控制循环外完成。

默认启用 Pi Plus 平地足部修正。它根据足底中点距地面（世界系 `z=0`）的高度估计支撑权重，累积有界的水平防滑偏移，通过阻尼逆运动学修正输出关节。流模型保持自身滚动历史。需要原始流模型输出，或场景地形不同，可以传入 `stabilize_feet=False`。

## 训练方法

实现采用本地论文 `docs/walk_like_this.pdf` 第 4 节和附录 A.1 的风格双采样、历史差分、朝向坐标系、风格注意力偏置和重叠滚动生成。扩散去噪替换为条件直线流匹配。

1. 从同一动作抽取 15 帧历史、随后 30 帧目标，以及独立抽取的 1–5 秒风格片段。长度允许时使用互不重叠的窗口。风格编码器每 4 帧取样，通过 register token 汇总姿态及其差分，输出单位范数嵌入。较长参考在推理时分段编码并按时长聚合。
2. 模型的输出变量包括根高度、横滚、俯仰和各铰链关节角。先减去从当前姿态向风格平均姿态过渡的基线，再将未来窗口投影到 8 个从零位移起步的低频正弦模态。截断高频也会削弱动作中的快速细节。
3. 令目标系数为 `a`，采样高斯噪声 `e` 和均匀流时间 `t`。在 `(1-t)e + ta` 上回归速度 `a-e`，按模态加权均方误差，联合训练生成器与风格编码器。条件包括风格、历史姿态及差分、未来平面路径和速度。
4. 训练时混合实测根路径与连接相同端点的平滑恒速转弯路径，使网络也能学习手柄式输入。推理时在 SE(2) 上积分实时速度，直接确定根平移和航向，再由流模型生成相应的全身姿态。关节角按 MJCF 的限位裁剪。根路径的跟踪精度由积分决定，步态质量和足底滑移需单独评估。

复现随附基线的训练命令如下。按文件名匹配后去重，以种子 42 抽取 3,600 个片段。目录中太短的片段会跳过。

```bash
python motion_flow/train.py --data data/pi_plus_motions \
  --pattern '*walk*.npz' 'crouch_ff*.npz' 'jog_ff*.npz' 'idle*.npz' \
  --max-clips 3600 --steps 10000 --batch-size 48 --device cuda \
  --out checkpoints/pi_plus_walk.pt
```

训练整个目录时省略 `--pattern` 和 `--max-clips`。加载器根据关节名对齐通道，根据 `body_names` 定位 `base_link`，支持标记为 `wxyz` 或 `xyzw` 的四元数。一个模型的训练片段和在线参考须具有同一骨架和采样率。当前权重针对 Pi Plus。

用 `--log-interval N` 设置打印间隔，默认每 100 步打印，第 1 步和最后一步也会打印。每行显示当前步数、总步数、完成百分比、当前批次的 `loss` 和自上次打印以来的 `mean_loss`。加载数据期间会显示已读取的文件数。

运行回归检查可使用 `python -m unittest discover -s tests -v`。

## 基线实测

最终权重使用 3,599 条有效动作（1,466,491 帧）训练了 10,000 步。下表的三条参考文件均未被本次训练抽中，同类动作可能出现在训练集中。三次生成均使用普通步行片段作为起始历史、随机种子 123 和前进 0.30 m/s 指令。每次生成 10 秒，统计后 8 秒。

| 风格参考 | 参考 / 生成平均根高度（m） | 平均关节姿态误差¹（rad） | 低位脚踝水平速度²（m/s） |
| --- | --- | --- | --- |
| 普通步行 `walk_ff_loop_360_003__A049` | 0.359 / 0.356 | 0.041 | 0.108 |
| 背手步行 `walk_hands_on_back_loop_001__A029` | 0.365 / 0.363 | 0.048 | 0.098 |
| 蹲姿步行 `crouch_ff_loop_360_003__A125` | 0.278 / 0.281 | 0.100 | 0.243 |

¹ 对每个关节分别计算生成和参考片段的时间均值，再取 22 个绝对差的平均值。该指标衡量静态姿态风格。步态节奏可在视频中检查。  
² 对两脚分别取脚踝高度最低四分之一的帧，统计世界系水平速度，用作足滑近似指标。关闭足部修正时三项分别为 0.165、0.192、0.307 m/s。蹲走在这个指令下仍有明显滑移。

在本机 Intel Core Ultra 9 275HX、PyTorch CPU 单线程、5 个 Euler 步条件下，预热后 144 次重规划帧耗时的 P95 为 4.54 ms（含单帧足部修正）。测量范围为 `step()`，风格编码、文件导出和底层控制器单独执行。变化指令测试覆盖前进、侧移、反向、转弯和停止。验证范围为运动学参考生成及 MuJoCo 正运动学，动力学跟踪和实机表现仍待验证。
