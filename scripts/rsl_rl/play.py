# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import os
import sys
from pathlib import Path

from packaging import version

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SOURCE_DIR = REPO_ROOT / "source" / "IRobot_wl"
if str(LOCAL_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE_DIR))

RSL_RL_VERSION = "3.1.2"


def _configure_temp_dir() -> Path:
    """Point temporary files to a writable workspace directory."""
    temp_dir = REPO_ROOT / ".tmp" / "isaaclab"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_str = str(temp_dir)
    os.environ["ISAACLAB_TMPDIR"] = temp_dir_str
    os.environ["TMPDIR"] = temp_dir_str
    os.environ["TMP"] = temp_dir_str
    os.environ["TEMP"] = temp_dir_str
    return temp_dir


def _check_and_preload_native_deps() -> str:
    """Validate RSL-RL version and preload native deps before Kit starts."""
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        print(
            f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
            f" and required version is: '{RSL_RL_VERSION}'."
        )
        raise SystemExit(1)

    # Preload native Python extensions before Omniverse Kit starts. On Windows, importing
    # these first avoids later DLL/access-violation failures during extension startup.
    import h5py  # noqa: F401
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: F401

    return installed_version


_configure_temp_dir()
installed_version = _check_and_preload_native_deps()

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--command_vx", type=float, default=None, help="Fixed forward velocity command for play [m/s].")
parser.add_argument("--command_yaw", type=float, default=None, help="Fixed yaw-rate command for play [rad/s].")
parser.add_argument("--command_height", type=float, default=None, help="Fixed base-height command for play [m].")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import IRobot_wl.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow
from wl_sequence import WlSequenceRunner

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    handle_deprecated_rsl_rl_cfg = None

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # handle deprecated configurations
    if handle_deprecated_rsl_rl_cfg is not None and agent_cfg.class_name != "WlSequenceRunner":
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.curriculum.recovery_stages = None
    if hasattr(env_cfg.curriculum, "command_levels_base_height"):
        env_cfg.curriculum.command_levels_base_height = None
    if hasattr(env_cfg.curriculum, "recovery_locomotion_commands"):
        env_cfg.curriculum.recovery_locomotion_commands = None

    if args_cli.command_vx is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_vx, args_cli.command_vx)
        if hasattr(env_cfg.commands.base_velocity, "rel_standing_envs"):
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if args_cli.command_yaw is not None:
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
        if hasattr(env_cfg.commands.base_velocity, "rel_standing_envs"):
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if args_cli.command_height is not None and hasattr(env_cfg.commands.base_velocity, "base_height_range"):
        env_cfg.commands.base_velocity.base_height_range = (args_cli.command_height, args_cli.command_height)

    keyboard_controller = None
    keyboard_command_cache = {"step": None, "raw": None, "obs": None}
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        policy_velocity_term = env_cfg.observations.policy.velocity_commands
        velocity_term_params = dict(getattr(policy_velocity_term, "params", {}) or {})
        default_height_command = float(
            args_cli.command_height if args_cli.command_height is not None else velocity_term_params.get("height_command", 0.25)
        )
        height_min, height_max = getattr(env_cfg.commands.base_velocity, "base_height_range", (0.19, 0.28))
        keyboard_height_command = {"value": max(float(height_min), min(default_height_command, float(height_max)))}
        original_velocity_obs_terms = {
            "policy": (
                env_cfg.observations.policy.velocity_commands.func,
                dict(getattr(env_cfg.observations.policy.velocity_commands, "params", {}) or {}),
            )
        }
        if hasattr(env_cfg.observations, "policy_history"):
            original_velocity_obs_terms["policy_history"] = (
                env_cfg.observations.policy_history.velocity_commands.func,
                dict(getattr(env_cfg.observations.policy_history.velocity_commands, "params", {}) or {}),
            )
        if hasattr(env_cfg.observations, "critic"):
            original_velocity_obs_terms["critic"] = (
                env_cfg.observations.critic.velocity_commands.func,
                dict(getattr(env_cfg.observations.critic.velocity_commands, "params", {}) or {}),
            )
        max_lin_vel_x = max(abs(v) for v in env_cfg.commands.base_velocity.ranges.lin_vel_x)
        heading_step_rate = 1.0
        config = Se2KeyboardCfg(
            # Use gentler commands than the full task range so each key press stays
            # close to the policy's typical training distribution.
            v_x_sensitivity=min(0.8, max_lin_vel_x),
            v_y_sensitivity=0.0,
            omega_z_sensitivity=heading_step_rate,
        )
        keyboard_controller = Se2Keyboard(config)
        keyboard_controller.reset()
        keyboard_controller.add_callback(
            "I",
            lambda: keyboard_height_command.__setitem__(
                "value", min(float(height_max), keyboard_height_command["value"] + 0.01)
            ),
        )
        keyboard_controller.add_callback(
            "K",
            lambda: keyboard_height_command.__setitem__(
                "value", max(float(height_min), keyboard_height_command["value"] - 0.01)
            ),
        )
        keyboard_controller.add_callback(
            "U",
            lambda: keyboard_height_command.__setitem__("value", max(float(height_min), min(default_height_command, float(height_max)))),
        )
        print(keyboard_controller)
        print(
            "[INFO] WL keyboard height controls: I raises target height, K lowers target height, U resets height "
            f"to {keyboard_height_command['value']:.3f} m."
        )
        keyboard_heading_target = {"value": None}

        def update_keyboard_command(env):
            step = int(getattr(env, "common_step_counter", 0))
            if keyboard_command_cache["step"] != step:
                keyboard_cmd = keyboard_controller.advance().to(env.device).unsqueeze(0)
                command = torch.zeros((env.num_envs, 3), device=env.device, dtype=keyboard_cmd.dtype)
                command[:, 0] = keyboard_cmd[:, 0]

                # Keep the command manager in sync so debug arrows, logging, and
                # any command-dependent terms see the same command as the policy.
                command_term = env.command_manager.get_term("base_velocity")
                if keyboard_heading_target["value"] is None:
                    keyboard_heading_target["value"] = env.scene["robot"].data.heading_w.clone()
                keyboard_heading_target["value"] = math_utils.wrap_to_pi(
                    keyboard_heading_target["value"] + keyboard_cmd[:, 2] * env.step_dt
                )
                if hasattr(command_term, "heading_target"):
                    command_term.heading_target[:] = keyboard_heading_target["value"]
                if hasattr(command_term, "base_height_command_b"):
                    command_term.base_height_command_b[:] = keyboard_height_command["value"]
                heading_error = math_utils.wrap_to_pi(
                    keyboard_heading_target["value"] - env.scene["robot"].data.heading_w
                )
                heading_control_stiffness = getattr(command_term.cfg, "heading_control_stiffness", 1.0)
                min_yaw, max_yaw = env_cfg.commands.base_velocity.ranges.ang_vel_z
                command[:, 2] = torch.clamp(heading_control_stiffness * heading_error, min=min_yaw, max=max_yaw)
                if hasattr(command_term, "vel_command_b"):
                    command_term.vel_command_b[:] = command
                if hasattr(command_term, "is_standing_env"):
                    command_term.is_standing_env[:] = torch.linalg.norm(command[:, [0, 2]], dim=1) < 1.0e-6
                if hasattr(command_term, "is_heading_env"):
                    command_term.is_heading_env[:] = True

                keyboard_command_cache["step"] = step
                keyboard_command_cache["raw"] = keyboard_cmd
                keyboard_command_cache["obs"] = {}

        def make_keyboard_command_obs(group_name):
            original_func, original_params = original_velocity_obs_terms[group_name]

            def keyboard_command_obs(env):
                update_keyboard_command(env)
                group_cache = keyboard_command_cache["obs"]
                if group_name not in group_cache:
                    group_cache[group_name] = original_func(env, **original_params)
                return group_cache[group_name]

            return keyboard_command_obs

        env_cfg.observations.policy.velocity_commands.func = make_keyboard_command_obs("policy")
        env_cfg.observations.policy.velocity_commands.params = {}
        if hasattr(env_cfg.observations, "policy_history"):
            env_cfg.observations.policy_history.velocity_commands.func = make_keyboard_command_obs("policy_history")
            env_cfg.observations.policy_history.velocity_commands.params = {}
        if hasattr(env_cfg.observations, "critic"):
            env_cfg.observations.critic.velocity_commands.func = make_keyboard_command_obs("critic")
            env_cfg.observations.critic.velocity_commands.params = {}

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        # Match train.py semantics: accept either a full checkpoint path or a
        # checkpoint filename relative to the selected experiment/run folder.
        if os.path.sep in args_cli.checkpoint or os.path.isabs(args_cli.checkpoint):
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.unwrapped._recovery_curriculum_stage = 4
    env.unwrapped._recovery_curriculum_iteration_offset = 0

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "WlSequenceRunner":
        runner = WlSequenceRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if agent_cfg.class_name == "WlSequenceRunner":
        pass
    elif version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Debug printing helper
    _robot = env.unwrapped.scene["robot"]
    _leg_joint_ids, _ = _robot.find_joints(
        ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"], preserve_order=True
    )
    _wheel_joint_ids, _ = _robot.find_joints(
        ["l_wheel_Joint", "r_wheel_Joint"], preserve_order=True
    )
    _leg_joint_ids = list(_leg_joint_ids)
    _wheel_joint_ids = list(_wheel_joint_ids)

    def _fmt(tensor, precision=3):
        return "[" + ", ".join(f"{v:.{precision}f}" for v in tensor.detach().cpu().tolist()) + "]"

    def print_debug():
        robot = _robot
        actions = env.unwrapped.action_manager.action
        torques = robot.data.applied_torque
        dof_vel = robot.data.joint_vel
        dof_pos = robot.data.joint_pos
        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b
        commands = env.unwrapped.command_manager.get_command("base_velocity")
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        base_height_cmd = getattr(command_term, "base_height_command_b", None)
        if base_height_cmd is None:
            base_height_cmd = torch.full((env.unwrapped.num_envs,), 0.235, device=robot.data.root_pos_w.device)
        keyboard_cmd = keyboard_command_cache["raw"] if keyboard_controller is not None else None

        wheel_vel = dof_vel[:, _wheel_joint_ids]
        vmc_wheel_vel = wheel_vel
        wheel_torque = torques[:, _wheel_joint_ids]
        vmc_wheel_torque = wheel_torque
        left_torque = torques[:, _leg_joint_ids[:2]]
        right_torque = torques[:, _leg_joint_ids[2:4]]

        from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state
        vmc_cfg = env.unwrapped.cfg.vmc_actions
        vmc_state = compute_vmc_state(
            dof_pos=dof_pos, dof_vel=dof_vel,
            leg_joint_indices=_leg_joint_ids,
            wheel_joint_indices=_wheel_joint_ids,
            l1=vmc_cfg.l1, l2=vmc_cfg.l2, offset=vmc_cfg.offset,
            theta1_offset=vmc_cfg.theta1_offset,
            theta2_offset=vmc_cfg.theta2_offset,
        )

        tp_cmd = torch.stack([actions[:, 0], actions[:, 3]], dim=1) * vmc_cfg.action_scale_tp
        delta_force_cmd = torch.stack([actions[:, 1], actions[:, 4]], dim=1) * vmc_cfg.action_scale_force
        total_force_cmd = delta_force_cmd + vmc_cfg.feedforward_force
        wheel_torque_cmd = torch.stack([actions[:, 2], actions[:, 5]], dim=1) * vmc_cfg.action_scale_wheel_torque

        w = 60
        print("#" * w)
        print(f" Step debug (Env0)")
        print(f"  Base lin vel [x,y,z]:      {_fmt(base_lin_vel[0])}")
        print(f"  Commands   [vx,vy,omega]:  {_fmt(commands[0])}")
        print(f"  Base height/cmd:           {robot.data.root_pos_w[0, 2].item():.3f} / {base_height_cmd[0].item():.3f}")
        if keyboard_cmd is not None:
            print(f"  Keyboard   [vx,vy,head]:   {_fmt(keyboard_cmd[0])}")
        print(f"  --- VMC task space ---")
        print(f"  theta0     [L, R]:         {_fmt(vmc_state['theta0'][0])}")
        print(f"  L0         [L, R]:         {_fmt(vmc_state['L0'][0])}")
        print(f"  Tp cmd     [L, R] Nm:      {_fmt(tp_cmd[0])}")
        print(f"  deltaF cmd [L, R] N:       {_fmt(delta_force_cmd[0])}")
        print(f"  total F    [L, R] N:       {_fmt(total_force_cmd[0])}")
        print(f"  --- Wheels ---")
        print(f"  joint wheel vel [L, R]:    {_fmt(wheel_vel[0])}")
        print(f"  wheel torque cmd [L, R]:   {_fmt(wheel_torque_cmd[0])}")
        print(f"  VMC wheel vel [L, R]:      {_fmt(vmc_wheel_vel[0])}")
        print(f"  joint torque  [L, R]:      {_fmt(wheel_torque[0])}")
        print(f"  VMC torque    [L, R]:      {_fmt(vmc_wheel_torque[0])}")
        print(f"  --- Leg torques ---")
        print(f"  left  [hip, knee]:         {_fmt(left_torque[0])}")
        print(f"  right [hip, knee]:         {_fmt(right_torque[0])}")
        print(f"  --- Actions (raw) ---")
        print(f"  left  [Tp, dF, wheel]:     {_fmt(actions[0, :3])}")
        print(f"  right [Tp, dF, wheel]:     {_fmt(actions[0, 3:6])}")
        print("#" * w)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    debug_print_every = int(1.0 / dt)  # print every 1 second (sim time)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if agent_cfg.class_name == "WlSequenceRunner":
                pass
            elif version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # periodic debug print
        if timestep % debug_print_every == 0:
            print_debug()

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
