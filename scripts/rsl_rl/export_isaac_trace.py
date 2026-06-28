#!/usr/bin/env python3
"""Export an Isaac Lab trace for WL flat-policy MuJoCo sim2sim alignment."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

from packaging import version

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
LOCAL_SOURCE_DIR = REPO_ROOT / "source" / "IRobot_wl"
if str(LOCAL_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE_DIR))

RSL_RL_VERSION = "3.1.2"


def _configure_temp_dir() -> None:
    temp_dir = REPO_ROOT / ".tmp" / "isaaclab"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ISAACLAB_TMPDIR"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)


def _check_and_preload_native_deps() -> str:
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        raise RuntimeError(f"rsl-rl-lib {installed_version} is older than required {RSL_RL_VERSION}")
    import h5py  # noqa: F401
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: F401

    return installed_version


_configure_temp_dir()
installed_version = _check_and_preload_native_deps()

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export WL Isaac trace for sim2sim alignment.")
parser.add_argument("--task", type=str, default="IRobot-WL-Velocity-VMC-Flat-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--out", type=str, default="sim2sim_mujoco/traces/isaac_flat_trace.jsonl")
parser.add_argument("--command-vx", type=float, default=1.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--height-command", type=float, default=0.235)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--init-theta0", type=float, default=None)
parser.add_argument("--init-l0", type=float, default=None)
parser.add_argument("--root-z", type=float, default=0.18)
parser.add_argument("--keep-reset-pose", action="store_true")
parser.add_argument("--isaac-usd-dir", type=str, default=str(WORKSPACE_ROOT / "sim2sim" / "isaac_usd_cache" / "wl"))
parser.add_argument("--force-usd-conversion", action="store_true")
parser.add_argument("--reuse-usd-path", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
if args_cli.checkpoint is None:
    args_cli.checkpoint = str(REPO_ROOT / "logs/rsl_rl/wl_vmc_flat/2026-06-25_11-41-46/model_15600.pt")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import IRobot_wl.tasks  # noqa: F401
from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state
from wl_sequence import WlSequenceRunner

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    handle_deprecated_rsl_rl_cfg = None


def _as_list(tensor) -> list[float]:
    return [float(v) for v in torch.as_tensor(tensor).detach().cpu().reshape(-1).tolist()]


def _set_fixed_commands(env, vx: float, yaw: float, height: float) -> None:
    command_term = env.command_manager.get_term("base_velocity")
    command = env.command_manager.get_command("base_velocity")
    command[:, 0] = vx
    command[:, 1] = 0.0
    command[:, 2] = yaw
    if hasattr(command_term, "vel_command_b"):
        command_term.vel_command_b[:] = command
    if hasattr(command_term, "base_height_command_b"):
        command_term.base_height_command_b[:] = height
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False


def _set_vmc_pose(env, theta0: float, l0: float) -> None:
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import inverse_kinematics

    robot = env.scene["robot"]
    cfg = env.cfg.vmc_actions
    leg_ids, _ = robot.find_joints(["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"], preserve_order=True)
    theta = torch.full((env.num_envs, 2), theta0, device=env.device)
    length = torch.full((env.num_envs, 2), l0, device=env.device)
    theta1, theta2 = inverse_kinematics(theta, length, cfg.l1, cfg.l2, cfg.offset)
    joint_pos = robot.data.joint_pos.clone()
    joint_pos[:, leg_ids[0]] = theta1[:, 0] - cfg.theta1_offset
    joint_pos[:, leg_ids[1]] = theta2[:, 0] - cfg.theta2_offset
    joint_pos[:, leg_ids[2]] = -(theta1[:, 1] - cfg.theta1_offset)
    joint_pos[:, leg_ids[3]] = -(theta2[:, 1] - cfg.theta2_offset)
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _set_root_state(env, root_z: float) -> None:
    robot = env.scene["robot"]
    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 0:3] = env.scene.env_origins
    root_pose[:, 2] += root_z
    root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))


def _zero_actions(env) -> None:
    zeros = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    env.action_manager.process_action(zeros)


def _refresh_sim(env) -> None:
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def _configure_usd_cache(env_cfg) -> None:
    spawn_cfg = env_cfg.scene.robot.spawn
    if args_cli.reuse_usd_path is not None:
        env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
            usd_path=str(Path(args_cli.reuse_usd_path)),
            visible=spawn_cfg.visible,
            semantic_tags=spawn_cfg.semantic_tags,
            copy_from_source=spawn_cfg.copy_from_source,
            mass_props=spawn_cfg.mass_props,
            deformable_props=spawn_cfg.deformable_props,
            rigid_props=spawn_cfg.rigid_props,
            collision_props=spawn_cfg.collision_props,
            activate_contact_sensors=spawn_cfg.activate_contact_sensors,
            scale=spawn_cfg.scale,
            articulation_props=spawn_cfg.articulation_props,
            fixed_tendons_props=spawn_cfg.fixed_tendons_props,
            spatial_tendons_props=spawn_cfg.spatial_tendons_props,
            joint_drive_props=spawn_cfg.joint_drive_props,
            visual_material_path=spawn_cfg.visual_material_path,
            visual_material=spawn_cfg.visual_material,
        )
        return
    usd_dir = Path(args_cli.isaac_usd_dir)
    usd_dir.mkdir(parents=True, exist_ok=True)
    spawn_cfg.usd_dir = str(usd_dir)
    spawn_cfg.usd_file_name = "wl.usd"
    spawn_cfg.force_usd_conversion = bool(args_cli.force_usd_conversion)


def _configure_local_flat_terrain(env_cfg) -> None:
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        },
    )
    env_cfg.scene.terrain.use_terrain_origins = False
    env_cfg.scene.terrain.visual_material = None


def _record(step: int, phase: str, obs_td, action, env) -> dict:
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("vmc")
    cfg = env.cfg.vmc_actions
    leg_ids, _ = robot.find_joints(["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"], preserve_order=True)
    wheel_ids, _ = robot.find_joints(["l_wheel_Joint", "r_wheel_Joint"], preserve_order=True)
    leg_ids = list(leg_ids)
    wheel_ids = list(wheel_ids)
    state = compute_vmc_state(
        dof_pos=robot.data.joint_pos,
        dof_vel=robot.data.joint_vel,
        leg_joint_indices=leg_ids,
        wheel_joint_indices=wheel_ids,
        l1=cfg.l1,
        l2=cfg.l2,
        offset=cfg.offset,
        theta1_offset=cfg.theta1_offset,
        theta2_offset=cfg.theta2_offset,
    )
    command = env.command_manager.get_command("base_velocity")
    command_term = env.command_manager.get_term("base_velocity")
    base_height = getattr(command_term, "base_height_command_b", torch.zeros(env.num_envs, device=env.device))
    raw = action_term.raw_actions
    processed = action_term.processed_actions
    delayed = getattr(action_term, "_delayed_actions", processed)
    torque_leg = torch.stack([delayed[:, 0], delayed[:, 3]], dim=1) * cfg.action_scale_tp
    force_leg = torch.stack([delayed[:, 1], delayed[:, 4]], dim=1) * cfg.action_scale_force + cfg.feedforward_force
    torque_wheel = torch.stack([delayed[:, 2], delayed[:, 5]], dim=1) * cfg.action_scale_wheel_torque
    return {
        "step": int(step),
        "phase": phase,
        "time": float(step * env.step_dt),
        "obs": _as_list(obs_td["policy"][0]),
        "history": _as_list(obs_td["policy_history"][0]),
        "raw_action": _as_list(raw[0]),
        "policy_action": _as_list(action[0]),
        "processed_action": _as_list(processed[0]),
        "delayed_action": _as_list(delayed[0]),
        "base_pos": _as_list(robot.data.root_pos_w[0]),
        "base_quat_wxyz": _as_list(robot.data.root_quat_w[0]),
        "base_lin_vel_body": _as_list(robot.data.root_lin_vel_b[0]),
        "base_ang_vel_body": _as_list(robot.data.root_ang_vel_b[0]),
        "projected_gravity_body": _as_list(robot.data.projected_gravity_b[0]),
        "command": _as_list(command[0]),
        "base_height_command": float(base_height[0]),
        "joint_pos": _as_list(robot.data.joint_pos[0]),
        "joint_vel": _as_list(robot.data.joint_vel[0]),
        "applied_torque": _as_list(robot.data.applied_torque[0]),
        "theta0": _as_list(state["theta0"][0]),
        "L0": _as_list(state["L0"][0]),
        "theta0_dot": _as_list(state["theta0_dot"][0]),
        "L0_dot": _as_list(state["L0_dot"][0]),
        "wheel_vel": _as_list(state["wheel_vel"][0]),
        "torque_leg": _as_list(torque_leg[0]),
        "force_leg": _as_list(force_leg[0]),
        "torque_wheel": _as_list(torque_wheel[0]),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "policy_history"):
        env_cfg.observations.policy_history.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False
    _configure_usd_cache(env_cfg)
    _configure_local_flat_terrain(env_cfg)
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    # Keep the original observation terms intact so IsaacLab's history buffers
    # are built correctly.  Fix the underlying command sampler instead.
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_vx, args_cli.command_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if hasattr(env_cfg.commands.base_velocity, "base_height_range"):
        env_cfg.commands.base_velocity.base_height_range = (args_cli.height_command, args_cli.height_command)
    for group_name in ("policy", "policy_history", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None and getattr(group, "velocity_commands", None) is not None:
            group.velocity_commands.params["height_command"] = args_cli.height_command

    if handle_deprecated_rsl_rl_cfg is not None and agent_cfg.class_name != "WlSequenceRunner":
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "WlSequenceRunner":
        runner = WlSequenceRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    _set_fixed_commands(env.unwrapped, args_cli.command_vx, args_cli.command_yaw, args_cli.height_command)
    if not args_cli.keep_reset_pose:
        _set_root_state(env.unwrapped, args_cli.root_z)
        if args_cli.init_theta0 is not None and args_cli.init_l0 is not None:
            _set_vmc_pose(env.unwrapped, args_cli.init_theta0, args_cli.init_l0)
    _zero_actions(env.unwrapped)
    _refresh_sim(env.unwrapped)
    env.unwrapped.observation_manager.reset()
    obs = env.get_observations()

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.unwrapped.device)
        f.write(json.dumps(_record(0, "pre", obs, zero_action, env.unwrapped), ensure_ascii=True) + "\n")
        for step in range(args_cli.steps):
            with torch.inference_mode():
                _set_fixed_commands(env.unwrapped, args_cli.command_vx, args_cli.command_yaw, args_cli.height_command)
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
                record = _record(step, "post", obs, action, env.unwrapped)
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                if torch.any(dones):
                    break
    print(f"wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
