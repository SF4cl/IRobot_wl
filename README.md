# IRobot_wl

`IRobot_wl` 是基于 Isaac Lab 的轮腿机器人训练工程。当前阶段先只维护平地和粗糙地形两个 VMC 速度跟踪任务，先把训练侧 observation、reward、action 和关节顺序完全理顺，再重新做 MuJoCo sim2sim。

## 当前维护的任务

```text
IRobot-WL-Velocity-VMC-Flat-v0
IRobot-WL-Velocity-VMC-Rough-v0
```

之前实验过的 recovery、stand、getup 等任务已经从当前注册入口移除。后续如果要重新做倒地自起，建议在 flat/rough 的关节顺序和 sim2sim 链路稳定之后，再单独恢复。

## 关节顺序约定

训练侧统一使用下面的标准顺序：

```python
leg_joint_names = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
wheel_joint_names = ["l_wheel_Joint", "r_wheel_Joint"]
```

凡是 VMC 几何、observation、reward、action 或诊断脚本里通过 `asset.find_joints(...)` 获取关节索引，都必须显式使用 `preserve_order=True`。不要依赖 articulation 内部顺序。

这点很关键：旧训练中 observation/reward 侧可能把 `[lf0, lf1, rf0, rf1]` 按 articulation 返回顺序错位解释，导致真实 `theta0/L0` 与 policy obs 里的腿角、腿长不一致。旧 checkpoint 因此不建议继续作为干净 sim2sim 的基准。

## 安装

默认工作目录：

```bash
cd /home/sf4/Workspace/rm/rl_wheel_legged/IRobot_wl/IRobot_wl
```

默认使用 Isaac Lab 训练环境：

```bash
conda activate rl_wheel_legged
pip install -e source/IRobot_wl
```

如果没有激活 shell，也可以用 `conda run -n rl_wheel_legged ...` 执行下面的命令。

## 重新训练

平地任务：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless
```

粗糙地形任务：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Rough-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless
```

小规模 smoke test：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 512 \
  --max_iterations 50 \
  --headless
```

## Play 和诊断

播放 flat 策略：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point
```

播放 rough 策略：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Rough-v0 \
  --agent rsl_rl_cfg_entry_point
```

查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

## MuJoCo sim2sim 前检查

新 checkpoint 用于 MuJoCo 前，建议先在 Isaac 侧导出 trace，对齐下面这些字段：

- policy 当前帧 observation
- policy history observation
- raw action 和 clipped action
- VMC 输出 torque
- `theta0/L0`
- base 姿态、速度和接触状态

只有 Isaac trace 内部确认 `theta0/L0` 和 policy obs 一致之后，再把同一 checkpoint 接到 `sim2sim_mujoco_rebuild`。旧的 `wl_vmc_flat/2026-06-25_11-41-46/model_15600.pt` 可以作为历史诊断材料，但不建议作为重新开始后的正式 sim2sim 基准。
