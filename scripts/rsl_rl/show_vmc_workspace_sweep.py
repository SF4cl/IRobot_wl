# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the WL VMC leg workspace in Isaac Sim.

The script fixes the base in the air and sweeps target virtual-leg length L0
and swing angle theta0 over nearly the full configured workspace.

Two visualization modes are available:

* set_state: write the IK joint state directly.  This is the cleanest geometry
  check and should cover the full workspace exactly.
* torque_pd: use a simple task-space PD controller to generate Tp/F actions,
  then let the current VMC action term map them to joint torques.

Examples:
    python scripts/rsl_rl/show_vmc_workspace_sweep.py --show
    python scripts/rsl_rl/show_vmc_workspace_sweep.py --show --mode torque_pd
    python scripts/rsl_rl/show_vmc_workspace_sweep.py --show --l0_count 17 --theta0_count 21 --steps_per_pose 20
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import ctypes
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Keep the Windows h5py workaround used by the local diagnostic scripts.  It is
# harmless on Linux because the directory normally does not exist.
_h5py_dll_dir = os.path.join(sys.prefix, "Lib", "site-packages", "h5py")
if os.path.isdir(_h5py_dll_dir):
    os.add_dll_directory(_h5py_dll_dir)
    ctypes.CDLL(os.path.join(_h5py_dll_dir, "hdf5.dll"))
    ctypes.CDLL(os.path.join(_h5py_dll_dir, "hdf5_hl.dll"))

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SOURCE_DIR = REPO_ROOT / "source" / "IRobot_wl"
if str(LOCAL_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Show a fixed-base sweep of WL VMC L0/theta0 workspace.")
parser.add_argument("--task", type=str, default="IRobot-WL-Velocity-VMC-Flat-v0")
parser.add_argument("--mode", choices=("set_state", "torque_pd"), default="set_state")
parser.add_argument("--l0_min", type=float, default=0.1219258562330587)
parser.add_argument("--l0_max", type=float, default=0.3006386827708927)
parser.add_argument("--theta0_min", type=float, default=-0.75)
parser.add_argument("--theta0_max", type=float, default=0.75)
parser.add_argument("--l0_count", type=int, default=15)
parser.add_argument("--theta0_count", type=int, default=17)
parser.add_argument("--steps_per_pose", type=int, default=30)
parser.add_argument("--cycles", type=int, default=1000000, help="Number of full workspace sweeps to repeat.")
parser.add_argument("--base_height", type=float, default=0.45)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--print_every_pose", type=int, default=10)
parser.add_argument("--same_target", action="store_true", default=True, help="Use the same target on both legs.")
parser.add_argument("--opposite_target", action="store_true", default=False, help="Use mirrored theta0 on right leg.")
parser.add_argument("--kp_l0", type=float, default=250.0)
parser.add_argument("--kd_l0", type=float, default=8.0)
parser.add_argument("--kp_theta", type=float, default=25.0)
parser.add_argument("--kd_theta", type=float, default=1.5)
parser.add_argument("--clip_force_actions", type=float, default=3.0)
parser.add_argument("--clip_tp_actions", type=float, default=3.0)
parser.add_argument("--feedforward_force", type=float, default=0.0)
parser.add_argument("--show", action="store_true", default=False, help="Show the simulator window.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.show:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg

import IRobot_wl.tasks  # noqa: F401
from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import (
    compute_vmc_state,
    inverse_kinematics,
    wrap_to_pi,
)


LEG_JOINT_NAMES = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
WHEEL_JOINT_NAMES = ["l_wheel_Joint", "r_wheel_Joint"]
WHEEL_BODY_NAMES = ["l_wheel_Link", "r_wheel_Link"]
HIP_JOINT_NAMES = ["lf0_Joint", "rf0_Joint"]


def _fmt(values: torch.Tensor, precision: int = 5) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values.detach().cpu().tolist()) + "]"


def _linspace(lo: float, hi: float, count: int) -> list[float]:
    if count <= 1:
        return [(lo + hi) * 0.5]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def _make_snake_targets() -> list[tuple[float, float]]:
    theta_values = _linspace(args_cli.theta0_min, args_cli.theta0_max, args_cli.theta0_count)
    l0_values = _linspace(args_cli.l0_min, args_cli.l0_max, args_cli.l0_count)
    targets: list[tuple[float, float]] = []
    for row, theta0 in enumerate(theta_values):
        row_l0 = l0_values if row % 2 == 0 else list(reversed(l0_values))
        for l0 in row_l0:
            targets.append((theta0, l0))
    return targets


def _parse_hip_origins_b(urdf_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
    origins = []
    for joint_name in HIP_JOINT_NAMES:
        origin = joints[joint_name].find("origin")
        origins.append([float(value) for value in origin.attrib["xyz"].split()])
    return torch.tensor(origins, device=device, dtype=dtype)


def _actual_leg_vector_from_links(robot, hip_origins_b: torch.Tensor) -> torch.Tensor:
    wheel_body_ids, wheel_body_names = robot.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)
    if wheel_body_names != WHEEL_BODY_NAMES:
        raise RuntimeError(f"Unexpected wheel body order: {wheel_body_names}")

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    hip_pos_w = root_pos_w.unsqueeze(1) + quat_apply(
        root_quat_w.unsqueeze(1).expand(-1, 2, -1),
        hip_origins_b.unsqueeze(0).expand(root_pos_w.shape[0], -1, -1),
    )
    wheel_pos_w = robot.data.body_pos_w[:, wheel_body_ids, :]
    return quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, 2, -1),
        wheel_pos_w - hip_pos_w,
    )


def _leg_state_from_vector(leg_vec_b: torch.Tensor, x_offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    sagittal_x = leg_vec_b[..., 0] + x_offset
    sagittal_down = -leg_vec_b[..., 2]
    leg_length = torch.sqrt(sagittal_x**2 + sagittal_down**2)
    leg_angle = torch.atan2(sagittal_down, sagittal_x) - torch.pi / 2
    return leg_length, leg_angle


def _target_tensors(env, theta0_scalar: float, l0_scalar: float) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    theta0 = torch.full((env.unwrapped.num_envs, 2), theta0_scalar, device=device, dtype=dtype)
    if args_cli.opposite_target:
        theta0[:, 1] = -theta0[:, 0]
    l0 = torch.full((env.unwrapped.num_envs, 2), l0_scalar, device=device, dtype=dtype)
    return theta0, l0


def _target_joint_state(env, theta0: torch.Tensor, l0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    cfg = env.unwrapped.cfg.actions.vmc
    theta1, theta2 = inverse_kinematics(theta0, l0, cfg.l1, cfg.l2, cfg.offset)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    leg_joint_ids, leg_joint_names = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
    wheel_joint_ids, wheel_joint_names = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
    if leg_joint_names != LEG_JOINT_NAMES:
        raise RuntimeError(f"Unexpected leg joint order: {leg_joint_names}")
    if wheel_joint_names != WHEEL_JOINT_NAMES:
        raise RuntimeError(f"Unexpected wheel joint order: {wheel_joint_names}")

    joint_pos[:, leg_joint_ids[0]] = theta1[:, 0] - cfg.theta1_offset
    joint_pos[:, leg_joint_ids[1]] = theta2[:, 0] - cfg.theta2_offset
    joint_pos[:, leg_joint_ids[2]] = -(theta1[:, 1] - cfg.theta1_offset)
    joint_pos[:, leg_joint_ids[3]] = cfg.theta2_offset - theta2[:, 1]
    joint_pos[:, wheel_joint_ids] = 0.0
    return joint_pos, joint_vel


def _write_fixed_base(env) -> None:
    robot = env.unwrapped.scene["robot"]
    root_pose = robot.data.root_pose_w.detach().clone()
    root_pose[:, 0] = 0.0
    root_pose[:, 1] = 0.0
    root_pose[:, 2] = args_cli.base_height
    root_pose[:, 3] = 1.0
    root_pose[:, 4:] = 0.0
    root_vel = torch.zeros_like(robot.data.root_vel_w.detach())
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_vel)


def _write_fixed_pose(env, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
    robot = env.unwrapped.scene["robot"]
    _write_fixed_base(env)
    robot.write_joint_state_to_sim(joint_pos.detach().clone(), joint_vel.detach().clone())
    robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_pos))
    if hasattr(env.unwrapped.scene, "write_data_to_sim"):
        env.unwrapped.scene.write_data_to_sim()


def _state(env) -> dict[str, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    cfg = env.unwrapped.cfg.actions.vmc
    leg_joint_ids, _ = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
    wheel_joint_ids, _ = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
    return compute_vmc_state(
        dof_pos=robot.data.joint_pos,
        dof_vel=robot.data.joint_vel,
        leg_joint_indices=leg_joint_ids,
        wheel_joint_indices=wheel_joint_ids,
        l1=cfg.l1,
        l2=cfg.l2,
        offset=cfg.offset,
        theta1_offset=cfg.theta1_offset,
        theta2_offset=cfg.theta2_offset,
        dt=env.unwrapped.step_dt,
    )


def _make_pd_action(env, theta0_target: torch.Tensor, l0_target: torch.Tensor) -> torch.Tensor:
    cfg = env.unwrapped.cfg.actions.vmc
    state = _state(env)
    theta_error = wrap_to_pi(theta0_target - state["theta0"])
    l0_error = l0_target - state["L0"]
    tp_cmd = args_cli.kp_theta * theta_error - args_cli.kd_theta * state["theta0_dot"]
    force_cmd = args_cli.kp_l0 * l0_error - args_cli.kd_l0 * state["L0_dot"] + args_cli.feedforward_force

    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action[:, 0] = tp_cmd[:, 0] / cfg.action_scale_tp
    action[:, 1] = (force_cmd[:, 0] - cfg.feedforward_force) / cfg.action_scale_force
    action[:, 3] = tp_cmd[:, 1] / cfg.action_scale_tp
    action[:, 4] = (force_cmd[:, 1] - cfg.feedforward_force) / cfg.action_scale_force
    action[:, [0, 3]].clamp_(-cfg.clip_tp_actions, cfg.clip_tp_actions)
    action[:, [1, 4]].clamp_(-cfg.clip_force_actions, cfg.clip_force_actions)
    return action


def _print_pose(env, pose_index: int, total_poses: int, theta0_target: torch.Tensor, l0_target: torch.Tensor) -> None:
    robot = env.unwrapped.scene["robot"]
    cfg = env.unwrapped.cfg.actions.vmc
    state = _state(env)
    urdf_path = env.unwrapped.cfg.scene.robot.spawn.asset_path
    hip_origins_b = _parse_hip_origins_b(urdf_path, robot.data.root_pos_w.device, robot.data.root_pos_w.dtype)
    leg_vec_b = _actual_leg_vector_from_links(robot, hip_origins_b)
    equivalent_l0, equivalent_theta0 = _leg_state_from_vector(leg_vec_b, x_offset=cfg.offset)
    print("\n" + "=" * 78)
    print(f"VMC workspace sweep pose {pose_index + 1}/{total_poses}, mode={args_cli.mode}")
    print(f"target theta0 [L,R]: {_fmt(theta0_target[0])}")
    print(f"target L0     [L,R]: {_fmt(l0_target[0])}")
    print(f"VMC theta0    [L,R]: {_fmt(state['theta0'][0])}")
    print(f"VMC L0        [L,R]: {_fmt(state['L0'][0])}")
    print(f"link theta0   [L,R]: {_fmt(equivalent_theta0[0])}")
    print(f"link L0       [L,R]: {_fmt(equivalent_l0[0])}")
    print(f"L0 error      [L,R]: {_fmt((state['L0'] - l0_target)[0])}")
    print(f"theta0 error  [L,R]: {_fmt(wrap_to_pi(state['theta0'] - theta0_target)[0])}")
    print(f"joint pos all:       {_fmt(robot.data.joint_pos[0])}")
    print("=" * 78)


def _configure_env(env_cfg) -> None:
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.scene.robot.spawn.fix_base = True
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.base_height)
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.observations.policy.height_scan = None
    env_cfg.observations.critic.height_scan = None

    env_cfg.actions.vmc.feedforward_force = args_cli.feedforward_force
    env_cfg.actions.vmc.clip_force_actions = args_cli.clip_force_actions
    env_cfg.actions.vmc.clip_tp_actions = args_cli.clip_tp_actions
    env_cfg.actions.vmc.action_scale_wheel_torque = 0.0
    env_cfg.actions.vmc.clip_wheel_actions = 0.0
    env_cfg.actions.vmc.randomize_vmc_gains = False
    env_cfg.actions.vmc.randomize_action_delay = False
    env_cfg.actions.vmc.action_delay_ms_range = (0.0, 0.0)

    if hasattr(env_cfg, "events"):
        env_cfg.events.randomize_rigid_body_material = None
        env_cfg.events.randomize_rigid_body_mass_base = None
        env_cfg.events.randomize_rigid_body_mass_others = None
        env_cfg.events.randomize_rigid_body_inertia = None
        env_cfg.events.randomize_com_positions = None
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.randomize_reset_joints = None
        env_cfg.events.randomize_actuator_gains = None
        env_cfg.events.randomize_push_robot = None
        env_cfg.events.randomize_reset_base.params["pose_range"] = {}
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {}


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_env(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")
    print(f"[INFO] mode={args_cli.mode}, fixed base height={args_cli.base_height}")
    print(f"[INFO] L0 range=[{args_cli.l0_min}, {args_cli.l0_max}], theta0 range=[{args_cli.theta0_min}, {args_cli.theta0_max}]")

    targets = _make_snake_targets()
    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    pose_counter = 0

    try:
        for _cycle in range(args_cli.cycles):
            for theta0_scalar, l0_scalar in targets:
                theta0_target, l0_target = _target_tensors(env, theta0_scalar, l0_scalar)
                joint_pos, joint_vel = _target_joint_state(env, theta0_target, l0_target)

                if args_cli.mode == "set_state":
                    _write_fixed_pose(env, joint_pos, joint_vel)
                    for _ in range(args_cli.steps_per_pose):
                        _write_fixed_pose(env, joint_pos, joint_vel)
                        env.step(zero_actions)
                        _write_fixed_pose(env, joint_pos, joint_vel)
                else:
                    # Start each target close to the previous simulated state, but
                    # keep the base fixed so this is a leg-controller diagnostic.
                    for _ in range(args_cli.steps_per_pose):
                        _write_fixed_base(env)
                        action = _make_pd_action(env, theta0_target, l0_target)
                        env.step(action)
                        _write_fixed_base(env)

                if pose_counter % max(args_cli.print_every_pose, 1) == 0:
                    _print_pose(env, pose_counter, len(targets), theta0_target, l0_target)
                pose_counter += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
