# WL Commands

本文档记录 `IRobot_wl` 中 WL 任务常用命令。

默认工作目录：

```bash
cd /home/sf4/Workspace/rm/rl_wheel_legged/IRobot_wl/IRobot_wl
```

默认环境：

```bash
conda activate rl_wheel_legged
```

## 1. 安装开发包

首次使用或代码结构有改动后，建议执行：

```bash
pip install -e source/IRobot_wl
```

## 2. 任务名

当前保留的 WL 任务名：

```bash
IRobot-WL-Velocity-VMC-Flat-v0
IRobot-WL-Velocity-VMC-Rough-v0
```

通常最常用的是：

```bash
IRobot-WL-Velocity-VMC-Flat-v0
```

## 3. VMC 动作语义

VMC policy 仍然输出 6 维 action，但腿部通道现在不再表示目标腿摆角 `theta0` 和目标腿长 `L0`，而是直接表示任务空间力/力矩：

```text
[Tp_l, deltaF_l, wheel_l, Tp_r, deltaF_r, wheel_r]
```

- `Tp`：腿摆角方向的任务空间力矩，经过 `action_scale_tp` 缩放后通过 VMC 雅可比映射到髋/膝关节力矩。
- `deltaF`：支撑腿轴向残差力，经过 `action_scale_force` 缩放。
- 最终轴向支撑力为 `deltaF + feedforward_force`。也就是说 policy 输出 0 时，腿部仍有配置里的前馈支撑力。
- `wheel`：轮速命令，语义保持不变。
- 旧的 target-PD 配置字段仍保留，主要用于兼容旧配置；当前腿部 action 路径不再用 policy action 生成目标腿角或目标腿长。

## 4. 开始训练

VMC Flat 训练：

```bash
python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless
```

VMC Rough 训练：

```bash
python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Rough-v0 \
  --agent rsl_rl_cfg_entry_point \
  --headless
```

## 5. 指定环境数训练

例如设置 `num_envs=512`：

```bash
python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 512 \
  --headless
```

## 6. 指定最大迭代数

例如只训练 500 轮：

```bash
python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --max_iterations 500 \
  --headless
```

## 7. 断点续训

指定某次 run 和 checkpoint：

```bash
python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --resume \
  --experiment_name wl_vmc_flat \
  --load_run 2026-06-09_12-00-00 \
  --checkpoint model_300.pt \
  --headless
```

自动续接某个实验目录下最新 checkpoint：

```bash
python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --resume \
  --experiment_name wl_vmc_flat \
  --checkpoint 'model_.*\.pt' \
  --headless
```

参数说明：

- `--experiment_name` 对应 `logs/rsl_rl/<experiment_name>`
- `--load_run` 对应某次训练目录名
- `--checkpoint` 对应 checkpoint 文件名或正则

## 8. Play 测试

播放某次训练的最新模型：

```bash
python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point
```

指定 run 和 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --load_run 2026-06-09_12-00-00 \
  --checkpoint model_300.pt
```

限制测试环境数：

```bash
python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 32
```

实时播放：

```bash
python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --real-time
```

键盘控制：

```bash
python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --keyboard
```

## 9. 录视频

训练时录视频：

```bash
python scripts/rsl_rl/train.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --video \
  --video_length 200 \
  --video_interval 2000
```

Play 时录视频：

```bash
python scripts/rsl_rl/play.py \
  --task IRobot-WL-Velocity-VMC-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --video \
  --video_length 300
```

## 10. TensorBoard

启动 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

如果只看某个实验：

```bash
tensorboard --logdir logs/rsl_rl/wl_vmc_flat --port 6006
```

浏览器打开：

```bash
http://localhost:6006
```

## 11. 日志和模型保存位置

训练日志默认保存在：

```bash
logs/rsl_rl/<experiment_name>/<timestamp_run_name>/
```

例如：

```bash
logs/rsl_rl/wl_vmc_flat/2026-06-09_12-00-00/
```

这个目录里通常有：

- `model_*.pt`
- `params/env.yaml`
- `params/agent.yaml`
- TensorBoard 事件文件
- `videos/`

查看最新 run：

```bash
ls -lt logs/rsl_rl/wl_vmc_flat
```

查看某次 run 下的模型：

```bash
ls logs/rsl_rl/wl_vmc_flat/2026-06-09_12-00-00
```

## 12. 常用排查

列出当前日志目录：

```bash
find logs/rsl_rl -maxdepth 3 -type f | tail -n 50
```

查看最新 checkpoint：

```bash
find logs/rsl_rl/wl_vmc_flat -maxdepth 2 -type f | grep 'model_.*\.pt' | sort
```

## 13. 推荐使用顺序

1. 先安装开发包
2. 跑 `VMC Flat` 训练
3. 用 `play.py` 看效果
4. 用 `--resume` 继续训练
5. 用 `tensorboard` 看 reward 和 episode length 曲线


python scripts\rsl_rl\train.py --task IRobot-WL-Velocity-VMC-Flat-v0 --agent rsl_rl_cfg_entry_point --num_envs 512 --max_iterations 200 --headless

scp -r -P 37023 root@connect.westc.seetacloud.com:/root/autodl-tmp/wl_workspace/IRobot_wl/logs/rsl_rl/wl_vmc_flat/2026-06-21_12-54-05 D:\rm\2026_code\rl\IRobot_wl\logs\rsl_rl\wl_vmc_flat