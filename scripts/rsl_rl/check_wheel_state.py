# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal wheel-state check for the WL VMC task.

The script applies a constant VMC wheel action and prints the wheel references,
joint velocities, torque command expected from the VMC wheel damping term, and
the wheel-link angular velocity projected onto the URDF joint axis.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check WL wheel action scaling, signs, velocity, and torque.")
parser.add_argument(
    "--task",
    type=str,
    default="IRobot-WL-Velocity-VMC-Flat-v0",
    help="Name of the VMC task to instantiate.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=200, help="Number of constant-action steps to run.")
parser.add_argument("--print_every", type=int, default=50, help="Print wheel state every N steps.")
parser.add_argument("--left_wheel_action", type=float, default=0.5, help="Raw action for left wheel.")
parser.add_argument("--right_wheel_action", type=float, default=0.5, help="Raw action for right wheel.")
parser.add_argument("--theta_action", type=float, default=0.0, help="Raw action for both leg angles.")
parser.add_argument("--l0_action", type=float, default=0.0, help="Raw action for both leg lengths.")
parser.add_argument("--wheel_damping", type=float, default=None, help="Temporarily override VMC wheel damping.")
parser.add_argument("--wheel_torque_limit", type=float, default=None, help="Temporarily override both VMC wheel torque limits.")
parser.add_argument("--sim_dt", type=float, default=None, help="Temporarily override physics simulation dt.")
parser.add_argument("--decimation", type=int, default=None, help="Temporarily override action decimation.")
parser.add_argument(
    "--print_before_step",
    action="store_true",
    default=False,
    help="Also print the state immediately before env.step() for printed steps.",
)
parser.add_argument(
    "--clean_reset",
    action="store_true",
    default=False,
    help="Disable startup/reset randomization terms for a cleaner one-env diagnostic.",
)
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


WHEEL_JOINT_NAMES = ["l_wheel_Joint", "r_wheel_Joint"]
WHEEL_BODY_NAMES = ["l_wheel_Link", "r_wheel_Link"]
WHEEL_PARENT_BODY_NAMES = ["lf1_Link", "rf1_Link"]


def _fmt(values: torch.Tensor, precision: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values.detach().cpu().tolist()) + "]"


def _rpy_to_matrix(rpy: list[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = torch.cos(torch.tensor(roll, device=device, dtype=dtype)), torch.sin(torch.tensor(roll, device=device, dtype=dtype))
    cp, sp = torch.cos(torch.tensor(pitch, device=device, dtype=dtype)), torch.sin(torch.tensor(pitch, device=device, dtype=dtype))
    cy, sy = torch.cos(torch.tensor(yaw, device=device, dtype=dtype)), torch.sin(torch.tensor(yaw, device=device, dtype=dtype))
    rx = torch.stack(
        [
            torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)]),
            torch.stack([torch.zeros_like(cr), cr, -sr]),
            torch.stack([torch.zeros_like(cr), sr, cr]),
        ]
    )
    ry = torch.stack(
        [
            torch.stack([cp, torch.zeros_like(cp), sp]),
            torch.stack([torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)]),
            torch.stack([-sp, torch.zeros_like(cp), cp]),
        ]
    )
    rz = torch.stack(
        [
            torch.stack([cy, -sy, torch.zeros_like(cy)]),
            torch.stack([sy, cy, torch.zeros_like(cy)]),
            torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)]),
        ]
    )
    return rz @ ry @ rx


def _parse_wheel_axes_parent(urdf_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Read wheel joint axes expressed in their parent-link frames."""
    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Unable to find robot URDF: {path}")

    joints = {joint.attrib["name"]: joint for joint in ET.parse(path).getroot().findall("joint")}
    axes = []
    for joint_name in WHEEL_JOINT_NAMES:
        joint = joints.get(joint_name)
        if joint is None:
            raise RuntimeError(f"Unable to find joint '{joint_name}' in {path}")
        axis = joint.find("axis")
        origin = joint.find("origin")
        axis_xyz = [float(value) for value in axis.attrib.get("xyz", "1 0 0").split()]
        rpy = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        axis_joint = torch.tensor(axis_xyz, device=device, dtype=dtype)
        axis_parent = _rpy_to_matrix(rpy, device, dtype) @ axis_joint
        axes.append(axis_parent / torch.linalg.norm(axis_parent))
    return torch.stack(axes, dim=0)


def _parse_wheel_axis_inertia(urdf_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Read wheel inertia about the URDF wheel joint axis."""
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    links = {link.attrib["name"]: link for link in root.findall("link")}

    inertias = []
    for joint_name in WHEEL_JOINT_NAMES:
        joint = joints[joint_name]
        child_name = joint.find("child").attrib["link"]
        axis = joint.find("axis")
        axis_xyz = [float(value) for value in axis.attrib.get("xyz", "1 0 0").split()]
        axis_tensor = torch.tensor(axis_xyz, device=device, dtype=dtype)
        axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor)

        inertia = links[child_name].find("inertial").find("inertia")
        values = {key: float(inertia.attrib[key]) for key in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]}
        inertia_matrix = torch.tensor(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ],
            device=device,
            dtype=dtype,
        )
        inertias.append(axis_tensor @ inertia_matrix @ axis_tensor)
    return torch.stack(inertias, dim=0)


def _wheel_axis_w(robot, urdf_path: str) -> torch.Tensor:
    parent_body_ids, parent_body_names = robot.find_bodies(WHEEL_PARENT_BODY_NAMES, preserve_order=True)
    if parent_body_names != WHEEL_PARENT_BODY_NAMES:
        raise RuntimeError(f"Unexpected wheel parent body order: {parent_body_names}")

    device = robot.data.root_pos_w.device
    dtype = robot.data.root_pos_w.dtype
    axes_parent = _parse_wheel_axes_parent(urdf_path, device, dtype)
    parent_quat_w = robot.data.body_quat_w[:, parent_body_ids, :]
    return quat_apply(
        parent_quat_w,
        axes_parent.unsqueeze(0).expand(parent_quat_w.shape[0], -1, -1),
    )


def _make_action(env) -> torch.Tensor:
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action[:, 0] = args_cli.theta_action
    action[:, 1] = args_cli.l0_action
    action[:, 2] = args_cli.left_wheel_action
    action[:, 3] = args_cli.theta_action
    action[:, 4] = args_cli.l0_action
    action[:, 5] = args_cli.right_wheel_action
    return action


def _print_wheel_state(env, action: torch.Tensor, step: int, phase: str) -> None:
    robot = env.unwrapped.scene["robot"]
    cfg = env.unwrapped.cfg.actions.vmc
    urdf_path = env.unwrapped.cfg.scene.robot.spawn.asset_path

    wheel_joint_ids, wheel_joint_names = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
    wheel_body_ids, wheel_body_names = robot.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)
    parent_body_ids, parent_body_names = robot.find_bodies(WHEEL_PARENT_BODY_NAMES, preserve_order=True)
    if wheel_joint_names != WHEEL_JOINT_NAMES:
        raise RuntimeError(f"Unexpected wheel joint order: {wheel_joint_names}")
    if wheel_body_names != WHEEL_BODY_NAMES:
        raise RuntimeError(f"Unexpected wheel body order: {wheel_body_names}")
    if parent_body_names != WHEEL_PARENT_BODY_NAMES:
        raise RuntimeError(f"Unexpected wheel parent body order: {parent_body_names}")

    wheel_action = torch.stack([action[:, 2], action[:, 5]], dim=1)
    wheel_vel_ref = wheel_action * cfg.action_scale_vel
    joint_pos = robot.data.joint_pos[:, wheel_joint_ids]
    joint_vel = robot.data.joint_vel[:, wheel_joint_ids]
    mirrored_joint_pos = torch.stack([joint_pos[:, 0], -joint_pos[:, 1]], dim=1)
    mirrored_joint_vel = torch.stack([joint_vel[:, 0], -joint_vel[:, 1]], dim=1)
    effort_target = robot.data.joint_effort_target[:, wheel_joint_ids]
    computed_torque = robot.data.computed_torque[:, wheel_joint_ids]
    applied_torque = robot.data.applied_torque[:, wheel_joint_ids]
    vmc_raw_torque = cfg.wheel_damping * (wheel_vel_ref - mirrored_joint_vel)
    configured_torque_limits = torch.as_tensor(cfg.torque_limits, device=joint_vel.device, dtype=joint_vel.dtype)
    vmc_clipped_physical_torque = torch.clamp(
        vmc_raw_torque,
        -configured_torque_limits[wheel_joint_ids],
        configured_torque_limits[wheel_joint_ids],
    )
    vmc_clipped_torque = torch.stack([vmc_clipped_physical_torque[:, 0], -vmc_clipped_physical_torque[:, 1]], dim=1)

    actuator = robot.actuators["wheel"]
    actuator_joint_indices = actuator.joint_indices.detach().cpu().tolist()
    actuator_local_ids = [actuator_joint_indices.index(idx) for idx in wheel_joint_ids]
    actuator_vel = joint_vel[:, actuator_local_ids]
    effort_limit = actuator.effort_limit[:, actuator_local_ids]
    saturation_effort = torch.as_tensor(actuator.cfg.saturation_effort, device=joint_vel.device, dtype=joint_vel.dtype)
    velocity_limit = actuator.velocity_limit[:, actuator_local_ids]
    vel_at_effort_limit = velocity_limit * (1.0 + effort_limit / saturation_effort)
    clipped_motor_vel = torch.clamp(actuator_vel, min=-vel_at_effort_limit, max=vel_at_effort_limit)
    motor_max_effort = torch.clamp(saturation_effort * (1.0 - clipped_motor_vel / velocity_limit), max=effort_limit)
    motor_min_effort = torch.clamp(saturation_effort * (-1.0 - clipped_motor_vel / velocity_limit), min=-effort_limit)
    wheel_axis_inertia = _parse_wheel_axis_inertia(urdf_path, joint_vel.device, joint_vel.dtype)
    physics_dt = getattr(env.unwrapped, "physics_dt", env.unwrapped.cfg.sim.dt)
    stable_damping = 2.0 * wheel_axis_inertia / physics_dt
    max_delta_vel = configured_torque_limits[wheel_joint_ids] / wheel_axis_inertia * physics_dt

    axis_w = _wheel_axis_w(robot, urdf_path)
    rel_ang_vel_w = robot.data.body_ang_vel_w[:, wheel_body_ids, :] - robot.data.body_ang_vel_w[:, parent_body_ids, :]
    projected_joint_vel = torch.sum(rel_ang_vel_w * axis_w, dim=-1)
    mirrored_projected_joint_vel = torch.stack([projected_joint_vel[:, 0], -projected_joint_vel[:, 1]], dim=1)

    wheel_pos_b = quat_apply_inverse(
        robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1),
        robot.data.body_pos_w[:, wheel_body_ids, :] - robot.data.root_pos_w.unsqueeze(1),
    )
    wheel_lin_vel_b = quat_apply_inverse(
        robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1),
        robot.data.body_lin_vel_w[:, wheel_body_ids, :],
    )

    print("\n" + "=" * 78)
    print(f"Wheel check at step {step} (env0, {phase})")
    print(f"URDF asset: {urdf_path}")
    print(f"wheel joints: {wheel_joint_names}, ids={wheel_joint_ids}")
    print(f"wheel bodies: {wheel_body_names}, ids={wheel_body_ids}")
    print(f"robot joint order: {robot.joint_names}")
    print(f"VMC torque limits full:        {_fmt(configured_torque_limits)}")
    print(f"VMC wheel torque limits:       {_fmt(configured_torque_limits[wheel_joint_ids])}")
    print(f"wheel axis inertia [kg*m2]:    {_fmt(wheel_axis_inertia)}")
    print(f"stable damping est 2I/dt:      {_fmt(stable_damping)}")
    print(f"max dvel/physics step [rad/s]: {_fmt(max_delta_vel)}")
    print(f"actuator wheel joint ids:      {actuator_joint_indices}")
    print(f"actuator effort limit [L, R]:  {_fmt(effort_limit[0])}")
    print(f"actuator velocity limit [L,R]: {_fmt(velocity_limit[0])}")
    print(f"wheel action raw [L, R]:       {_fmt(wheel_action[0])}")
    print(f"wheel vel ref [rad/s] [L, R]:  {_fmt(wheel_vel_ref[0])}")
    print(f"joint pos [rad] [L, R]:        {_fmt(joint_pos[0])}")
    print(f"mirrored pos [rad] [L, R]:     {_fmt(mirrored_joint_pos[0])}")
    print(f"joint vel [rad/s] [L, R]:      {_fmt(joint_vel[0])}")
    print(f"mirrored vel [rad/s] [L, R]:   {_fmt(mirrored_joint_vel[0])}")
    print(f"axis-proj vel [rad/s] [L, R]:  {_fmt(projected_joint_vel[0])}")
    print(f"mirrored axis vel [L, R]:      {_fmt(mirrored_projected_joint_vel[0])}")
    print(f"vel projection error [L, R]:   {_fmt((joint_vel - projected_joint_vel)[0])}")
    print(f"VMC physical torque [L, R]:    {_fmt(vmc_raw_torque[0])}")
    print(f"VMC joint torque [L, R]:       {_fmt(vmc_clipped_torque[0])}")
    print(f"joint effort target [L, R]:    {_fmt(effort_target[0])}")
    print(f"computed torque [Nm] [L, R]:   {_fmt(computed_torque[0])}")
    print(f"motor min effort [Nm] [L, R]:  {_fmt(motor_min_effort[0])}")
    print(f"motor max effort [Nm] [L, R]:  {_fmt(motor_max_effort[0])}")
    print(f"applied torque [Nm] [L, R]:    {_fmt(applied_torque[0])}")
    print(f"target error [Nm] [L, R]:      {_fmt((effort_target - vmc_clipped_torque)[0])}")
    print(f"actuator error [Nm] [L, R]:    {_fmt((applied_torque - computed_torque)[0])}")
    print(f"wheel axis world [x,y,z] L:    {_fmt(axis_w[0, 0])}")
    print(f"wheel axis world [x,y,z] R:    {_fmt(axis_w[0, 1])}")
    print(f"wheel pos base [x,y,z] L:      {_fmt(wheel_pos_b[0, 0])}")
    print(f"wheel pos base [x,y,z] R:      {_fmt(wheel_pos_b[0, 1])}")
    print(f"wheel lin vel base [x,y,z] L:  {_fmt(wheel_lin_vel_b[0, 0])}")
    print(f"wheel lin vel base [x,y,z] R:  {_fmt(wheel_lin_vel_b[0, 1])}")
    print(f"base lin vel body [x,y,z]:     {_fmt(robot.data.root_lin_vel_b[0])}")
    print(f"base ang vel body [x,y,z]:     {_fmt(robot.data.root_ang_vel_b[0])}")
    print("=" * 78)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.debug_vis = False
    if args_cli.sim_dt is not None:
        env_cfg.sim.dt = args_cli.sim_dt
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
        env_cfg.sim.render_interval = args_cli.decimation
    if args_cli.wheel_damping is not None:
        env_cfg.actions.vmc.wheel_damping = args_cli.wheel_damping
    if args_cli.wheel_torque_limit is not None:
        torque_limits = list(env_cfg.actions.vmc.torque_limits)
        wheel_joint_ids = [4, 5]
        for joint_id in wheel_joint_ids:
            torque_limits[joint_id] = args_cli.wheel_torque_limit
        env_cfg.actions.vmc.torque_limits = torque_limits
    if args_cli.clean_reset and hasattr(env_cfg, "events"):
        env_cfg.events.randomize_rigid_body_mass_base = None
        env_cfg.events.randomize_rigid_body_mass_others = None
        env_cfg.events.randomize_com_positions = None
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.randomize_actuator_gains = None
        env_cfg.events.randomize_reset_base.params["pose_range"] = {}
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {}
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")

    env.reset()
    action = _make_action(env)
    _print_wheel_state(env, action, step=0, phase="after reset, before first action")

    for step in range(1, args_cli.steps + 1):
        should_print = step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps
        if args_cli.print_before_step and should_print:
            _print_wheel_state(env, action, step=step, phase="before env.step")
        with torch.inference_mode():
            env.step(action)
        if should_print:
            _print_wheel_state(env, action, step=step, phase="after env.step")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
