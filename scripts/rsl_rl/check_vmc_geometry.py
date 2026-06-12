# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal VMC geometry check for the WL robot.

This script starts the VMC environment, reads the initial joint state, computes
L0/theta0 with the VMC forward kinematics, and compares them against the real
hip-to-wheel vector from the simulated articulation link poses.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check WL VMC L0/theta0 against simulated link geometry.")
parser.add_argument(
    "--task",
    type=str,
    default="IRobot-WL-Velocity-VMC-Flat-v0",
    help="Name of the VMC task to instantiate.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=0, help="Number of zero-action steps before printing.")
parser.add_argument("--print_every", type=int, default=50, help="Print geometry every N zero-action steps.")
parser.add_argument("--show", action="store_true", default=False, help="Show the simulator window.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
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
from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state


LEG_JOINT_NAMES = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
WHEEL_JOINT_NAMES = ["l_wheel_Joint", "r_wheel_Joint"]
WHEEL_BODY_NAMES = ["l_wheel_Link", "r_wheel_Link"]

HIP_JOINT_NAMES = ["lf0_Joint", "rf0_Joint"]


def _fmt(values: torch.Tensor, precision: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values.detach().cpu().tolist()) + "]"


def _parse_hip_origins_b(urdf_path: str) -> torch.Tensor:
    """Read left/right hip joint origins from the URDF used by the environment."""
    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Unable to find robot URDF: {path}")

    joints = {joint.attrib["name"]: joint for joint in ET.parse(path).getroot().findall("joint")}
    origins = []
    for joint_name in HIP_JOINT_NAMES:
        joint = joints.get(joint_name)
        if joint is None:
            raise RuntimeError(f"Unable to find joint '{joint_name}' in {path}")
        origin = joint.find("origin")
        if origin is None or "xyz" not in origin.attrib:
            raise RuntimeError(f"Joint '{joint_name}' in {path} does not have an origin xyz")
        origins.append([float(value) for value in origin.attrib["xyz"].split()])
    return torch.tensor(origins)


def _leg_state_from_vector(leg_vec_b: torch.Tensor, x_offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    sagittal_x = leg_vec_b[..., 0] + x_offset
    sagittal_down = -leg_vec_b[..., 2]
    leg_length = torch.sqrt(sagittal_x**2 + sagittal_down**2)
    leg_angle = torch.atan2(sagittal_down, sagittal_x) - torch.pi / 2
    return leg_length, leg_angle


def _actual_leg_vector_from_links(robot, hip_origins_b: torch.Tensor) -> torch.Tensor:
    """Compute real hip-to-wheel leg length and sagittal leg angle from link poses."""
    wheel_body_ids, wheel_body_names = robot.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)
    if wheel_body_names != WHEEL_BODY_NAMES:
        raise RuntimeError(f"Unexpected wheel body order: {wheel_body_names}")

    device = robot.data.root_pos_w.device
    dtype = robot.data.root_pos_w.dtype
    hip_origins_b = hip_origins_b.to(device=device, dtype=dtype)

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    hip_pos_w = root_pos_w.unsqueeze(1) + quat_apply(
        root_quat_w.unsqueeze(1).expand(-1, 2, -1),
        hip_origins_b.unsqueeze(0).expand(root_pos_w.shape[0], -1, -1),
    )
    wheel_pos_w = robot.data.body_pos_w[:, wheel_body_ids, :]

    leg_vec_b = quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, 2, -1),
        wheel_pos_w - hip_pos_w,
    )
    return leg_vec_b


def _print_geometry(env, step: int) -> None:
    robot = env.unwrapped.scene["robot"]
    cfg = env.unwrapped.cfg.actions.vmc
    urdf_path = env.unwrapped.cfg.scene.robot.spawn.asset_path
    hip_origins_b = _parse_hip_origins_b(urdf_path)

    leg_joint_ids, leg_joint_names = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
    wheel_joint_ids, wheel_joint_names = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
    if leg_joint_names != LEG_JOINT_NAMES:
        raise RuntimeError(f"Unexpected leg joint order: {leg_joint_names}")
    if wheel_joint_names != WHEEL_JOINT_NAMES:
        raise RuntimeError(f"Unexpected wheel joint order: {wheel_joint_names}")

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
    leg_vec_b = _actual_leg_vector_from_links(robot, hip_origins_b)
    physical_l0, physical_theta0 = _leg_state_from_vector(leg_vec_b)
    equivalent_l0, equivalent_theta0 = _leg_state_from_vector(leg_vec_b, x_offset=cfg.offset)

    l0_err = state["L0"] - equivalent_l0
    theta_err = state["theta0"] - equivalent_theta0

    print("\n" + "=" * 78)
    print(f"VMC geometry check at step {step} (env0)")
    print(f"URDF asset: {urdf_path}")
    print(f"hip origins [L, R] in base frame: {_fmt(hip_origins_b[0])} {_fmt(hip_origins_b[1])}")
    print(f"VMC x offset: {cfg.offset:.9f}")
    print(f"joint_pos {leg_joint_names}: {_fmt(robot.data.joint_pos[0, leg_joint_ids])}")
    print(f"theta1 mirrored [L, R]:     {_fmt(state['theta1'][0])}")
    print(f"theta2 mirrored [L, R]:     {_fmt(state['theta2'][0])}")
    print("-" * 78)
    print(f"VMC FK L0      [L, R]:      {_fmt(state['L0'][0])}")
    print(f"link equiv L0  [L, R]:      {_fmt(equivalent_l0[0])}")
    print(f"L0 error       [L, R]:      {_fmt(l0_err[0])}")
    print(f"physical L0    [L, R]:      {_fmt(physical_l0[0])}")
    print("-" * 78)
    print(f"VMC FK theta0  [L, R]:      {_fmt(state['theta0'][0])}")
    print(f"link equiv theta[L, R]:     {_fmt(equivalent_theta0[0])}")
    print(f"theta0 error   [L, R]:      {_fmt(theta_err[0])}")
    print(f"physical theta [L, R]:      {_fmt(physical_theta0[0])}")
    print("-" * 78)
    print(f"hip->wheel in base frame [x,y,z] L: {_fmt(leg_vec_b[0, 0])}")
    print(f"hip->wheel in base frame [x,y,z] R: {_fmt(leg_vec_b[0, 1])}")
    print("=" * 78)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")

    env.reset()
    _print_geometry(env, step=0)

    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for step in range(1, args_cli.steps + 1):
        with torch.inference_mode():
            env.step(zero_actions)
        if step % args_cli.print_every == 0 or step == args_cli.steps:
            _print_geometry(env, step=step)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
