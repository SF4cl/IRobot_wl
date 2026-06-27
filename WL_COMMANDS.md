# WL Commands

本文档记录 `IRobot_wl` 当前建议使用的 WL 训练、播放和诊断命令。

当前阶段只专注两个任务：

```text
IRobot-WL-Velocity-VMC-Flat-v0
IRobot-WL-Velocity-VMC-Rough-v0
```

recovery、stand、getup 等旧实验任务已经从当前注册入口移除，暂时不要继续用旧 checkpoint 做 sim2sim 基准。

## 1. 默认路径和环境

```bash
cd /home/sf4/Workspace/rm/rl_wheel_legged/IRobot_wl/IRobot_wl
conda activate rl_wheel_legged
```

未激活 shell 时，可以把命令写成：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py ...
```

安装开发包：

```bash
pip install -e source/IRobot_wl
```

## 2. 关节顺序标准

VMC 相关代码统一使用：

```python
leg_joint_names = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
wheel_joint_names = ["l_wheel_Joint", "r_wheel_Joint"]
```

所有 `asset.find_joints(...)` 只要用于 VMC 几何、observation、reward、action 或诊断，都必须显式加：

```python
preserve_order=True
```

不要依赖 Isaac articulation 的内部返回顺序。之前发现过真实 `theta0/L0` 是 `0/0.235`，但 policy obs 里的 leg angle/length 不一致的问题，这类现象很像关节顺序错位导致。

## 3. VMC 动作语义

VMC policy 输出 6 维 action：

```text
[Tp_l, deltaF_l, wheel_l, Tp_r, deltaF_r, wheel_r]
```

- `Tp_l/Tp_r`: 左右腿摆角方向任务空间力矩。
- `deltaF_l/deltaF_r`: 左右腿轴向残差支撑力。
- `wheel_l/wheel_r`: 左右轮控制输出。
- 腿部 `Tp/deltaF` 会经过 action scale，再通过 VMC 雅可比映射为髋/膝关节 torque。
- policy 输出 0 时，腿部仍可能通过配置里的 feedforward force 获得基础支撑力。

## 4. 训练命令

Flat 正式训练：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless
```

Rough 正式训练：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Rough-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless
```

Flat smoke test：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 512 \
  --max_iterations 50 \
  --headless
```

Rough smoke test：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Rough-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 512 \
  --max_iterations 50 \
  --headless
```

## 5. 断点续训

继续某个 flat run：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --resume \
  --experiment_name wl_vmc_flat \
  --load_run <run_name> \
  --checkpoint <checkpoint_name> \
  --headless
```

继续某个 rough run：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Rough-v0 \
  --agent rsl_rl_cfg_entry_point \
  --resume \
  --experiment_name wl_vmc_rough \
  --load_run <run_name> \
  --checkpoint <checkpoint_name> \
  --headless
```

注意：旧的 `wl_vmc_flat/2026-06-25_11-41-46/model_15600.pt` 是关节顺序修正前的历史模型，不建议在它上面继续训练作为新基准。建议从当前代码重新训练。

## 6. Play 测试

播放 flat 最新模型：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point
```

播放 rough 最新模型：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Rough-v0 \
  --agent rsl_rl_cfg_entry_point
```

指定 run 和 checkpoint：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --load_run <run_name> \
  --checkpoint <checkpoint_name>
```

实时播放：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --real-time
```

键盘控制：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --keyboard
```

## 7. 录视频

训练时录视频：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --video \
  --video_length 200 \
  --video_interval 2000 \
  --headless
```

Play 时录视频：

```bash
conda run -n rl_wheel_legged python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --video \
  --video_length 300
```

## 8. TensorBoard

查看所有 WL 训练：

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

只看 flat：

```bash
tensorboard --logdir logs/rsl_rl/wl_vmc_flat --port 6006
```

只看 rough：

```bash
tensorboard --logdir logs/rsl_rl/wl_vmc_rough --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

## 9. 日志和模型位置

训练日志默认保存在：

```bash
logs/rsl_rl/<experiment_name>/<timestamp_run_name>/
```

常见文件：

```text
model_*.pt
params/env.yaml
params/agent.yaml
events.out.tfevents.*
videos/
```

查看 flat 最新 run：

```bash
ls -lt logs/rsl_rl/wl_vmc_flat
```

查看 rough 最新 run：

```bash
ls -lt logs/rsl_rl/wl_vmc_rough
```

查找 checkpoint：

```bash
find logs/rsl_rl/wl_vmc_flat -maxdepth 2 -type f | grep 'model_.*\.pt' | sort
```

## 10. Isaac trace 导出建议

新训练完成后，先导出 Isaac trace，再接 MuJoCo。trace 中至少要核对：

- 当前帧 observation
- history observation
- raw action 和 clipped action
- VMC torque
- `theta0/L0`
- base roll/pitch/yaw、速度和接触状态

目标是先确认 Isaac 内部的真实 `theta0/L0` 与 policy obs 中的腿角、腿长一致。确认之后，再用同一个 checkpoint 做 MuJoCo 对齐。

## 11. 推荐流程

1. 安装开发包。
2. 跑 flat smoke test，确认任务能正常启动。
3. 从当前代码重新训练 flat。
4. 用 `play.py` 看 Isaac 内是否能稳定站立和速度跟踪。
5. 导出 Isaac trace，确认 observation、VMC 几何、action、torque 顺序一致。
6. 进入 `sim2sim_mujoco_rebuild` 做 MuJoCo 单步对齐和闭环测试。
7. flat 稳定后，再训练 rough。
