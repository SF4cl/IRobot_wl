# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize a fixed-base WL pose generated from VMC target theta0/L0.

This is a small diagnostic for checking whether equal left/right VMC targets
produce a visually symmetric leg posture.  It fixes the base in the air, disables
VMC torques, writes the IK joint state every step, and prints both the target and
measured VMC state.

Example:

    python scripts/rsl_rl/show_vmc_target_pose.py --show --theta0 0.0 --l0 0.17 --base_height 0.45
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Ensure h5py's bundled HDF5 DLLs are loaded before Isaac Sim's conflicting copies.
# Isaac Sim ships its own hdf5.dll (for rt sensors) that gets loaded first
# via Kit's DLL search path. Once loaded, Windows won't load another copy from a
# different path, and h5py's .pyd files fail because the exports don't match.
import ctypes
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


parser = argparse.ArgumentParser(description="Show the WL pose for a target VMC theta0/L0.")
parser.add_argument("--task", type=str, default="IRobot-WL-Velocity-VMC-Flat-v0")
parser.add_argument("--theta0", type=float, default=0.0, help="Target virtual leg angle [rad].")
parser.add_argument("--l0", type=float, default=0.17, help="Target virtual leg length [m].")
parser.add_argument("--base_height", type=float, default=0.45, help="Fixed base height [m].")
parser.add_argument("--steps", type=int, default=100000, help="Number of render/sim steps to hold the pose.")
parser.add_argument("--print_every", type=int, default=120, help="Print state every N steps.")
parser.add_argument("--num_envs", type=int, default=1)
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
)


LEG_JOINT_NAMES = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
WHEEL_JOINT_NAMES = ["l_wheel_Joint", "r_wheel_Joint"]
WHEEL_BODY_NAMES = ["l_wheel_Link", "r_wheel_Link"]
HIP_JOINT_NAMES = ["lf0_Joint", "rf0_Joint"]


def _fmt(values: torch.Tensor, precision: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values.detach().cpu().tolist()) + "]"


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


def _make_target_joint_state(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    cfg = env.unwrapped.cfg.actions.vmc
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype

    theta0 = torch.full((env.unwrapped.num_envs, 2), args_cli.theta0, device=device, dtype=dtype)
    l0 = torch.full((env.unwrapped.num_envs, 2), args_cli.l0, device=device, dtype=dtype)
    theta1, theta2 = inverse_kinematics(theta0, l0, cfg.l1, cfg.l2, cfg.offset)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    leg_joint_ids, leg_joint_names = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
    wheel_joint_ids, wheel_joint_names = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
    if leg_joint_names != LEG_JOINT_NAMES:
        raise RuntimeError(f"Unexpected leg joint order: {leg_joint_names}")
    if wheel_joint_names != WHEEL_JOINT_NAMES:
        raise RuntimeError(f"Unexpected wheel joint order: {wheel_joint_names}")

    # Map mirrored VMC leg-frame angles back to physical joint coordinates.
    joint_pos[:, leg_joint_ids[0]] = theta1[:, 0] - cfg.theta1_offset
    joint_pos[:, leg_joint_ids[1]] = theta2[:, 0] - cfg.theta2_offset
    joint_pos[:, leg_joint_ids[2]] = -(theta1[:, 1] - cfg.theta1_offset)
    joint_pos[:, leg_joint_ids[3]] = cfg.theta2_offset - theta2[:, 1]
    joint_pos[:, wheel_joint_ids] = 0.0
    return joint_pos, joint_vel, theta1, theta2


def _write_fixed_pose(env, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
    robot = env.unwrapped.scene["robot"]
    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 2] = args_cli.base_height
    # identity orientation in Isaac Lab quaternion order [w, x, y, z]
    root_pose[:, 3] = 1.0
    root_pose[:, 4:] = 0.0
    root_vel = torch.zeros_like(robot.data.root_vel_w)

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_pos))
    if hasattr(env.unwrapped.scene, "write_data_to_sim"):
        env.unwrapped.scene.write_data_to_sim()


def _print_state(env, target_theta1: torch.Tensor, target_theta2: torch.Tensor, step: int) -> None:
    robot = env.unwrapped.scene["robot"]
    cfg = env.unwrapped.cfg.actions.vmc
    urdf_path = env.unwrapped.cfg.scene.robot.spawn.asset_path
    leg_joint_ids, _ = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
    wheel_joint_ids, _ = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)

    state = compute_vmc_state(
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
    hip_origins_b = _parse_hip_origins_b(urdf_path, robot.data.root_pos_w.device, robot.data.root_pos_w.dtype)
    leg_vec_b = _actual_leg_vector_from_links(robot, hip_origins_b)
    physical_l0, physical_theta0 = _leg_state_from_vector(leg_vec_b)
    equivalent_l0, equivalent_theta0 = _leg_state_from_vector(leg_vec_b, x_offset=cfg.offset)

    print("\n" + "=" * 78)
    print(f"VMC target pose at step {step}")
    print(f"target theta0={args_cli.theta0:.6f}, target L0={args_cli.l0:.6f}, fixed base z={args_cli.base_height:.3f}")
    print(f"target theta1 mirrored [L, R]: {_fmt(target_theta1[0])}")
    print(f"target theta2 mirrored [L, R]: {_fmt(target_theta2[0])}")
    print(f"physical joint pos {LEG_JOINT_NAMES}: {_fmt(robot.data.joint_pos[0, leg_joint_ids])}")
    print("-" * 78)
    print(f"VMC theta0 [L, R]:            {_fmt(state['theta0'][0])}")
    print(f"VMC L0     [L, R]:            {_fmt(state['L0'][0])}")
    print(f"equiv link theta0 [L, R]:     {_fmt(equivalent_theta0[0])}")
    print(f"equiv link L0     [L, R]:     {_fmt(equivalent_l0[0])}")
    print(f"physical hip-wheel theta [L,R]: {_fmt(physical_theta0[0])}")
    print(f"physical hip-wheel L0    [L,R]: {_fmt(physical_l0[0])}")
    print("-" * 78)
    print(f"hip->wheel vector base L [x,y,z]: {_fmt(leg_vec_b[0, 0])}")
    print(f"hip->wheel vector base R [x,y,z]: {_fmt(leg_vec_b[0, 1])}")
    print("=" * 78)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Keep this as a clean visual/geometry diagnostic, not a locomotion rollout.
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.scene.robot.spawn.fix_base = True
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.base_height)
    env_cfg.actions.vmc.feedforward_force = 0.0
    env_cfg.actions.vmc.action_scale_tp = 0.0
    env_cfg.actions.vmc.action_scale_force = 0.0
    env_cfg.actions.vmc.action_scale_wheel_torque = 0.0
    env_cfg.actions.vmc.randomize_action_delay = False
    env_cfg.actions.vmc.action_delay_ms_range = (0.0, 0.0)

    if hasattr(env_cfg, "events"):
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    joint_pos, joint_vel, target_theta1, target_theta2 = _make_target_joint_state(env)
    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    _write_fixed_pose(env, joint_pos, joint_vel)
    _print_state(env, target_theta1, target_theta2, step=0)

    for step in range(1, args_cli.steps + 1):
        with torch.inference_mode():
            _write_fixed_pose(env, joint_pos, joint_vel)
            env.step(zero_actions)
            _write_fixed_pose(env, joint_pos, joint_vel)
        if step % args_cli.print_every == 0 or step == args_cli.steps:
            _print_state(env, target_theta1, target_theta2, step=step)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
