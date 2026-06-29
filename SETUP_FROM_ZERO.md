# IRobot_wl 从零配置环境指南

这份文档按 Windows + PowerShell + Conda + Isaac Lab 的方式写。默认工程根目录为：

```powershell
D:\rm\2026_code\rl
```

如果你的工程放在别的位置，把下面命令里的路径替换掉即可。

## 1. 目录结构

推荐把 `IsaacLab` 和 `IRobot_wl` 放在同一个工作目录下：

```text
D:\rm\2026_code\rl
├─ IsaacLab
├─ IRobot_wl
├─ sim2sim
└─ condaenvs
```

`IRobot_wl` 是基于 Isaac Lab 的外部任务包。训练脚本在：

```text
IRobot_wl\scripts\rsl_rl\train.py
IRobot_wl\scripts\rsl_rl\play.py
IRobot_wl\scripts\rsl_rl\export_onnx.py
```

当前主要任务名：

```text
IRobot-WL-Velocity-VMC-Flat-v0
IRobot-WL-Velocity-VMC-Rough-v0
```

## 2. 基础软件

需要提前安装：

- NVIDIA 显卡驱动，建议使用较新的 Studio/Game Ready Driver。
- Miniconda 或 Anaconda。
- Git。
- Visual Studio Code，可选但推荐。
- Windows 长路径支持，可选但推荐。Isaac Sim/Isaac Lab 路径很深，没开长路径时 pip 安装可能失败。

检查显卡驱动：

```powershell
nvidia-smi
```

检查 Conda：

```powershell
conda --version
```

## 3. 创建 Conda 环境

Isaac Lab 当前仓库对应 Isaac Sim 5.1.0，推荐 Python 3.11。

```powershell
conda create -n isaacsim510 python=3.11 -y
conda activate isaacsim510
python -m pip install --upgrade pip
```

如果 PowerShell 里 `conda activate` 不能用，先执行：

```powershell
conda init powershell
```

然后重新打开 PowerShell。

如果每次打开 PowerShell 都看到类似：

```text
profile.ps1 cannot be loaded because running scripts is disabled
```

可以用管理员或当前用户权限执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. 安装 Isaac Sim

推荐用 pip 安装 Isaac Sim 5.1.0：

```powershell
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

安装 CUDA 版 PyTorch。Isaac Lab 5.1 文档推荐的 Windows x86_64 组合是：

```powershell
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

验证 Isaac Sim 能启动 Python 包：

```powershell
python -c "import isaacsim; print('Isaac Sim import OK')"
```

第一次启动 Isaac Sim/Isaac Lab 会比较慢，因为会初始化 Omniverse Kit 缓存。

## 5. 安装 Isaac Lab

进入 Isaac Lab 目录：

```powershell
cd D:\rm\2026_code\rl\IsaacLab
```

安装 Isaac Lab 及常用 RL 依赖：

```powershell
.\isaaclab.bat --install
```

如果只想明确安装 rsl_rl 相关依赖，也可以执行：

```powershell
.\isaaclab.bat -p -m pip install rsl-rl-lib==3.1.2
```

验证 Isaac Lab：

```powershell
.\isaaclab.bat -p -c "import isaaclab; import isaaclab_tasks; print('Isaac Lab import OK')"
```

也可以跑一个 Isaac Lab 自带小测试：

```powershell
.\isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py --headless
```

能正常启动并退出，说明 Isaac Sim + Isaac Lab 主环境基本可用。

## 6. 安装 IRobot_wl

进入 `IRobot_wl`：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl
```

用 editable 模式安装任务包：

```powershell
pip install -e source\IRobot_wl
```

补齐训练脚本需要的包：

```powershell
pip install tensorboard onnx onnxruntime h5py pyyaml
```

检查包能否导入：

```powershell
python -c "import IRobot_wl; import IRobot_wl.tasks; print('IRobot_wl import OK')"
```

如果这里失败，优先检查：

- 当前是否已经 `conda activate isaacsim510`。
- 是否在 `D:\rm\2026_code\rl\IRobot_wl` 下执行了 `pip install -e source\IRobot_wl`。
- `source\IRobot_wl\IRobot_wl` 目录是否存在。

## 7. 注册任务检查

可以用一个很小的 headless smoke test 检查任务是否能被 Gym/Isaac Lab 找到：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl

python scripts\rsl_rl\train.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --num_envs 64 `
  --max_iterations 2 `
  --headless
```

正常现象：

- Isaac Sim 会启动。
- 终端会打印 `Simulation App Startup Complete`。
- 日志会写入 `logs\rsl_rl\wl_vmc_flat\<timestamp>\`。
- 跑完 2 个 iteration 后退出。

如果报 `rsl-rl-lib` 版本太低，执行：

```powershell
pip install -U rsl-rl-lib==3.1.2
```

## 8. 正式训练

Flat 任务：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl

python scripts\rsl_rl\train.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --num_envs 4096 `
  --max_iterations 5000 `
  --headless
```

Rough 任务：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl

python scripts\rsl_rl\train.py `
  --task IRobot-WL-Velocity-VMC-Rough-v0 `
  --num_envs 4096 `
  --max_iterations 5000 `
  --headless
```

如果显存不够，先把 `--num_envs` 降到 `2048`、`1024` 或 `512`。

## 9. 从 checkpoint 续训

日志和模型默认保存在：

```text
IRobot_wl\logs\rsl_rl\<experiment_name>\<run_name>\
```

例如：

```text
logs\rsl_rl\wl_vmc_flat\2026-06-29_01-51-02_stage4_height_stronger_from_7774\model_8898.pt
```

续训命令：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl

python scripts\rsl_rl\train.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --num_envs 4096 `
  --max_iterations 5000 `
  --resume `
  --load_run <run_name> `
  --checkpoint <checkpoint_name> `
  --headless
```

例子：

```powershell
python scripts\rsl_rl\train.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --num_envs 4096 `
  --max_iterations 5000 `
  --resume `
  --load_run 2026-06-29_01-51-02_stage4_height_stronger_from_7774 `
  --checkpoint model_8898.pt `
  --run_name continue_from_8898 `
  --headless
```

如果想强制从 recovery curriculum 的某个阶段开始，例如直接从平跑 stage4 开始：

```powershell
python scripts\rsl_rl\train.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --num_envs 4096 `
  --max_iterations 5000 `
  --resume `
  --load_run 2026-06-29_01-51-02_stage4_height_stronger_from_7774 `
  --checkpoint model_8898.pt `
  --run_name stage4_continue_from_8898 `
  --recovery_start_stage 4 `
  --headless
```

`--recovery_start_stage` 可选值：

```text
0, 1, 2, 3, 4
```

## 10. 播放策略

播放 latest 或指定 checkpoint：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl

python scripts\rsl_rl\play.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --num_envs 16
```

指定 run 和 checkpoint：

```powershell
python scripts\rsl_rl\play.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --load_run <run_name> `
  --checkpoint <checkpoint_name> `
  --num_envs 16
```

实时播放：

```powershell
python scripts\rsl_rl\play.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --load_run <run_name> `
  --checkpoint <checkpoint_name> `
  --num_envs 1 `
  --real-time
```

键盘控制：

```powershell
python scripts\rsl_rl\play.py `
  --task IRobot-WL-Velocity-VMC-Flat-v0 `
  --load_run <run_name> `
  --checkpoint <checkpoint_name> `
  --keyboard
```

## 11. TensorBoard

启动 TensorBoard：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl
tensorboard --logdir logs\rsl_rl --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

只看 flat：

```powershell
tensorboard --logdir logs\rsl_rl\wl_vmc_flat --port 6006
```

## 12. 导出 ONNX

导出训练好的 checkpoint：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl

python scripts\rsl_rl\export_onnx.py `
  --run_dir logs\rsl_rl\wl_vmc_flat\<run_name> `
  --checkpoint <checkpoint_name>
```

指定输出路径：

```powershell
python scripts\rsl_rl\export_onnx.py `
  --run_dir logs\rsl_rl\wl_vmc_flat\<run_name> `
  --checkpoint <checkpoint_name> `
  --output exported\policy.onnx
```

导出的 ONNX 可以放到 `sim2sim\policies` 下用于 MuJoCo sim2sim。

## 13. MuJoCo sim2sim 环境

MuJoCo 建议单独使用轻量环境，不要和 Isaac Sim 环境混在一起。

创建环境：

```powershell
conda create -n mujoco python=3.11 -y
conda activate mujoco
pip install mujoco numpy onnxruntime torch pyyaml
```

运行示例：

```powershell
cd D:\rm\2026_code\rl

python sim2sim\run_wl_policy_mujoco.py `
  --xml sim2sim\wl\wl.xml `
  --backend onnx `
  --policy-onnx sim2sim\policies\<policy_name>.onnx `
  --viewer `
  --duration 300 `
  --print-every 1.0 `
  --orientation-preset upright `
  --base-z 0.25 `
  --no-wheel-contact-only `
  --stabilize-joints
```

如果你已经有本地 Python：

```powershell
D:\rm\2026_code\rl\condaenvs\mujoco\python.exe sim2sim\run_wl_policy_mujoco.py ...
```

## 14. 常见问题

### 14.1 找不到任务

错误类似：

```text
Environment IRobot-WL-Velocity-VMC-Flat-v0 doesn't exist
```

处理：

```powershell
cd D:\rm\2026_code\rl\IRobot_wl
pip install -e source\IRobot_wl
python -c "import IRobot_wl.tasks; print('task import OK')"
```

同时确认训练脚本没有被移动。`train.py` 里会自动把 `source\IRobot_wl` 加进 `sys.path`，但 editable 安装仍然推荐保留。

### 14.2 rsl-rl-lib 版本不对

训练脚本要求：

```text
rsl-rl-lib >= 3.1.2
```

修复：

```powershell
pip install -U rsl-rl-lib==3.1.2
```

### 14.3 PowerShell profile.ps1 报错

如果每次命令前都有：

```text
profile.ps1 cannot be loaded because running scripts is disabled
```

修复：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

这通常不影响 Python 运行，但会让输出很乱。

### 14.4 py_compile 写 __pycache__ 被拒绝

如果看到：

```text
WinError 5 拒绝访问 __pycache__
```

这通常是 Windows 权限或文件占用问题。可以用只读语法检查替代：

```powershell
python -c "import ast, pathlib; ast.parse(pathlib.Path('source/IRobot_wl/IRobot_wl/tasks/manager_based/locomotion/velocity/mdp/rewards.py').read_text(encoding='utf-8')); print('syntax OK')"
```

### 14.5 显存不够

优先降低并行环境数量：

```powershell
--num_envs 2048
--num_envs 1024
--num_envs 512
```

也可以先用：

```powershell
--headless
```

避免渲染占用资源。

### 14.6 第一次启动很慢

Isaac Sim 第一次启动会建立 shader/cache/extscache，可能需要几分钟。只要没有明确报错，可以等一会儿。

## 15. 推荐从零验证顺序

1. `nvidia-smi` 正常。
2. `conda activate isaacsim510` 正常。
3. `python -c "import isaacsim"` 正常。
4. `IsaacLab\isaaclab.bat -p -c "import isaaclab"` 正常。
5. `pip install -e IRobot_wl\source\IRobot_wl` 完成。
6. `python -c "import IRobot_wl.tasks"` 正常。
7. 跑 `--num_envs 64 --max_iterations 2 --headless` smoke test。
8. 跑正式 flat 训练。
9. 用 TensorBoard 看 reward、diagnostics。
10. 导出 ONNX，再做 MuJoCo sim2sim。

