"""Virtual Model Control (VMC) for wheel-legged robots.

Reference: Wheel-Legged-Gym (https://github.com/leggedrobotics/legged_gym)

VMC enables task-space control by converting desired end-effector forces and
torques into joint torques via the Jacobian transpose. The task space is defined
by leg angle (theta0) and leg length (L0) in the sagittal plane.
"""

from __future__ import annotations

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle tensor to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def forward_kinematics(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    l1: float,
    l2: float,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute task-space state (leg length L0, leg angle theta0) from joint angles.

    The forward kinematics map from joint space (theta1=hip, theta2=knee) to
    task space where:
      - L0: distance from hip to end-effector (wheel center)
      - theta0: angle of the leg in the sagittal plane (0 = straight down)

    Args:
        theta1: Hip joint angles, shape (num_envs, num_legs).
        theta2: Knee joint angles, shape (num_envs, num_legs).
        l1: Length of the thigh link [m].
        l2: Length of the calf link [m].
        offset: Hip offset from the body center [m].

    Returns:
        L0: Leg length, shape (num_envs, num_legs).
        theta0: Leg angle, shape (num_envs, num_legs).
    """
    end_x = offset + l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
    end_y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)
    L0 = torch.sqrt(end_x**2 + end_y**2)
    theta0 = torch.arctan2(end_y, end_x) - torch.pi / 2
    return L0, theta0


def compute_vmc_state(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    leg_joint_indices: list[int],
    wheel_joint_indices: list[int],
    l1: float,
    l2: float,
    offset: float,
    theta1_offset: float = 0.0,
    theta2_offset: float = torch.pi / 2,
    dt: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Compute the mirrored leg-frame state used by the original WL-Gym VMC."""

    theta1 = torch.stack(
        [dof_pos[:, leg_joint_indices[0]] + theta1_offset, -dof_pos[:, leg_joint_indices[2]] + theta1_offset],
        dim=1,
    )
    theta2 = torch.stack(
        [dof_pos[:, leg_joint_indices[1]] + theta2_offset, -dof_pos[:, leg_joint_indices[3]] + theta2_offset],
        dim=1,
    )
    theta1_dot = torch.stack(
        [dof_vel[:, leg_joint_indices[0]], -dof_vel[:, leg_joint_indices[2]]], dim=1
    )
    theta2_dot = torch.stack(
        [dof_vel[:, leg_joint_indices[1]], -dof_vel[:, leg_joint_indices[3]]], dim=1
    )

    L0, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)
    L0_fwd, theta0_fwd = forward_kinematics(theta1 + theta1_dot * dt, theta2 + theta2_dot * dt, l1, l2, offset)
    L0_dot = (L0_fwd - L0) / dt
    theta0_dot = wrap_to_pi(theta0_fwd - theta0) / dt

    # Wheel task coordinates follow the physical joint axes. Static wheel-only
    # checks should show a positive wheel reference driving positive joint/VMC
    # wheel velocity on both sides.
    wheel_pos = dof_pos[:, wheel_joint_indices]
    wheel_vel = dof_vel[:, wheel_joint_indices]

    return {
        "theta1": theta1,
        "theta2": theta2,
        "theta1_dot": theta1_dot,
        "theta2_dot": theta2_dot,
        "L0": L0,
        "theta0": theta0,
        "L0_dot": L0_dot,
        "theta0_dot": theta0_dot,
        "wheel_pos": wheel_pos,
        "wheel_vel": wheel_vel,
    }


def inverse_kinematics(
    theta0: torch.Tensor,
    L0: torch.Tensor,
    l1: float,
    l2: float,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute joint angles (theta1=hip, theta2=knee) from task-space state.

    Given desired leg angle theta0 and leg length L0, solve for the hip and knee
    joint angles using the law of cosines.

    Args:
        theta0: Desired leg angle, shape (num_envs, num_legs). 0 = straight down.
        L0: Desired leg length from hip to end-effector, shape (num_envs, num_legs).
        l1: Length of the thigh link [m].
        l2: Length of the calf link [m].
        offset: Hip offset from the body center [m].

    Returns:
        theta1: Hip joint angles, shape (num_envs, num_legs).
        theta2: Knee joint angles, shape (num_envs, num_legs).
    """
    gamma = theta0 + torch.pi / 2  # convert to coordinate frame
    target_x = L0 * torch.cos(gamma) - offset
    target_y = L0 * torch.sin(gamma)
    target_len = torch.sqrt(target_x**2 + target_y**2)
    target_len = torch.clamp(target_len, min=abs(l1 - l2) + 1e-4, max=l1 + l2 - 1e-4)

    # Law of cosines for the knee angle
    cos_beta = (l1**2 + l2**2 - target_len**2) / (2.0 * l1 * l2)
    cos_beta = torch.clamp(cos_beta, -1.0 + 1e-6, 1.0 - 1e-6)
    beta = torch.acos(cos_beta)  # angle between thigh and calf, 0 = fully folded

    # Knee joint angle: theta2 = pi - beta (0 = straight leg, positive = backward bend)
    theta2 = torch.pi - beta

    # Hip joint angle from geometry, accounting for the fixed hip x-offset.
    alpha = torch.atan2(l2 * torch.sin(theta2), l1 + l2 * torch.cos(theta2))
    theta1 = torch.atan2(target_y, target_x) - alpha

    return theta1, theta2


def vmc_torques(
    theta0: torch.Tensor,
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    L0: torch.Tensor,
    F_leg: torch.Tensor,
    T_leg: torch.Tensor,
    l1: float,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute joint torques from task-space forces using the VMC Jacobian transpose.

    Given desired leg force F_leg (along leg axis, positive = push) and leg torque
    T_leg (about the virtual leg angle theta0), compute the corresponding joint
    torques for hip (T1) and knee (T2).  The task coordinates are the same as
    ``forward_kinematics()``: [L0, theta0].

    Args:
        theta0: Leg angle in task space, shape (num_envs, num_legs).
        theta1: Hip joint angle, shape (num_envs, num_legs).
        theta2: Knee joint angle, shape (num_envs, num_legs).
        L0: Leg length, shape (num_envs, num_legs).
        F_leg: Desired leg force along the leg axis [N], shape (num_envs, num_legs).
        T_leg: Desired leg torque about the hip [Nm], shape (num_envs, num_legs).
        l1: Length of the thigh link [m].
        l2: Length of the calf link [m].

    Returns:
        T1: Hip joint torque, shape (num_envs, num_legs).
        T2: Knee joint torque, shape (num_envs, num_legs).
    """
    gamma = theta0 + torch.pi / 2
    L0_safe = torch.clamp(torch.nan_to_num(L0, nan=0.05, posinf=0.05, neginf=0.05), min=0.05)

    # Jacobian transpose for task coordinates [L0, theta0].
    # tau = dL0/dq * F_leg + dtheta0/dq * T_leg
    dL_dtheta1 = l1 * torch.sin(gamma - theta1) - l2 * torch.sin(theta1 + theta2 - gamma)
    dL_dtheta2 = -l2 * torch.sin(theta1 + theta2 - gamma)

    dtheta_dtheta1 = (
        l1 * torch.cos(gamma - theta1) + l2 * torch.cos(theta1 + theta2 - gamma)
    ) / L0_safe
    dtheta_dtheta2 = l2 * torch.cos(theta1 + theta2 - gamma) / L0_safe

    T1 = dL_dtheta1 * F_leg + dtheta_dtheta1 * T_leg
    T2 = dL_dtheta2 * F_leg + dtheta_dtheta2 * T_leg
    T1 = torch.nan_to_num(T1, nan=0.0, posinf=0.0, neginf=0.0)
    T2 = torch.nan_to_num(T2, nan=0.0, posinf=0.0, neginf=0.0)

    return T1, T2


def compute_vmc_action(
    actions: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    leg_joint_indices: list[int],
    wheel_joint_indices: list[int],
    l1: float,
    l2: float,
    offset: float,
    theta1_offset: float,
    theta2_offset: float,
    theta0_offset: float,
    kp_theta: float,
    kd_theta: float,
    kp_l0: float,
    kd_l0: float,
    l0_offset: float,
    l0_min: float,
    l0_max: float,
    feedforward_force: float,
    action_scale_tp: float,
    action_scale_force: float,
    action_scale_wheel_torque: float,
    torque_limits: torch.Tensor,
    torque_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Compute joint torques from VMC task-space actions.

    This function takes policy actions in task-space coordinates and converts them
    to joint torques using Virtual Model Control. The action space is 6-dimensional
    for a bipedal robot with 2 legs:
      [Tp_left, deltaF_left, wheel_torque_left, Tp_right, deltaF_right, wheel_torque_right]

    Leg actions (Tp, deltaF) are converted through VMC Jacobian transpose.
    Wheel actions are direct torque commands — the policy learns its own
    velocity control, bypassing the damping-based PD of the old design.

    Args:
        actions: Policy actions in task space, shape (num_envs, 6).
        dof_pos: Current joint positions, shape (num_envs, num_dof).
        dof_vel: Current joint velocities, shape (num_envs, num_dof).
        leg_joint_indices: Indices of leg joints [hip_left, knee_left, hip_right, knee_right].
        wheel_joint_indices: Indices of wheel joints [wheel_left, wheel_right].
        l1: Thigh link length [m].
        l2: Calf link length [m].
        offset: Hip offset from body center [m].
        kp_theta: Legacy angle PD proportional gain, kept for config compatibility.
        kd_theta: Legacy angle PD derivative gain, kept for config compatibility.
        kp_l0: Legacy length PD proportional gain, kept for config compatibility.
        kd_l0: Legacy length PD derivative gain, kept for config compatibility.
        l0_offset: Legacy default leg length offset, kept for config compatibility.
        feedforward_force: Gravity compensation force [N].
        action_scale_tp: Scale for leg swing torque actions [Nm].
        action_scale_force: Scale for residual axial force actions [N].
        action_scale_wheel_torque: Scale for wheel direct torque actions [Nm].
        torque_limits: Joint torque limits, shape (num_dof,).

    Returns:
        torques: Joint torques, shape (num_envs, num_dof).
    """
    num_envs = actions.shape[0]
    num_dof = dof_pos.shape[1]

    # --- Parse task-space force/torque actions ---
    torque_leg = torch.stack([actions[:, 0], actions[:, 3]], dim=1) * action_scale_tp
    force_leg_delta = torch.stack([actions[:, 1], actions[:, 4]], dim=1) * action_scale_force
    force_leg = force_leg_delta + feedforward_force

    # Direct wheel torque: policy action * scale → Nm
    torque_wheel = torch.stack([actions[:, 2], actions[:, 5]], dim=1) * action_scale_wheel_torque

    state = compute_vmc_state(
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        leg_joint_indices=leg_joint_indices,
        wheel_joint_indices=wheel_joint_indices,
        l1=l1,
        l2=l2,
        offset=offset,
        theta1_offset=theta1_offset,
        theta2_offset=theta2_offset,
    )
    theta1 = state["theta1"]
    theta2 = state["theta2"]
    L0 = state["L0"]
    theta0 = state["theta0"]
    wheel_vel = state["wheel_vel"]

    # --- VMC: task-space force/torque to joint torques ---
    T1, T2 = vmc_torques(theta0, theta1, theta2, L0, force_leg, torque_leg, l1, l2)

    # --- Assemble full torque vector ---
    torques = torch.zeros(num_envs, num_dof, device=actions.device, dtype=actions.dtype)

    # Left leg
    torques[:, leg_joint_indices[0]] = T1[:, 0]  # left hip
    torques[:, leg_joint_indices[1]] = T2[:, 0]  # left knee

    # Right leg (negate due to mirror convention)
    torques[:, leg_joint_indices[2]] = -T1[:, 1]  # right hip
    torques[:, leg_joint_indices[3]] = -T2[:, 1]  # right knee

    # Wheels: direct torque from policy action
    torques[:, wheel_joint_indices[0]] = torque_wheel[:, 0]  # left wheel
    torques[:, wheel_joint_indices[1]] = torque_wheel[:, 1]  # right wheel

    # Apply motor torque scale (domain randomization, matching WL-Gym)
    torques = torques * torque_scale
    torques = torch.nan_to_num(torques, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip to torque limits
    torques = torch.clamp(torques, -torque_limits, torque_limits)

    return torques


class WLVMCAction(ActionTerm):
    """VMC action term that maps task-space force commands to joint torques.

    The leg channels command swing torque and residual axial force directly.
    The old leg PD gain fields are retained for config compatibility but are not
    used by this action path.
    """

    cfg: "WLVMCActionCfg"

    def __init__(self, cfg: "WLVMCActionCfg", env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._delayed_actions = torch.zeros_like(self._raw_actions)
        self._previous_actions = torch.zeros_like(self._raw_actions)
        self._previous_previous_actions = torch.zeros_like(self._raw_actions)
        self._last_torques = torch.zeros((self.num_envs, self._asset.num_joints), device=self.device)

        self._leg_joint_ids, _ = self._asset.find_joints(cfg.leg_joint_names, preserve_order=True)
        self._wheel_joint_ids, _ = self._asset.find_joints(cfg.wheel_joint_names, preserve_order=True)

        # --- Domain randomization buffers (per environment, fixed at init) ---
        self._kp_theta = torch.full((self.num_envs, 1), cfg.kp_theta, device=self.device)
        self._kd_theta = torch.full((self.num_envs, 1), cfg.kd_theta, device=self.device)
        self._kp_l0 = torch.full((self.num_envs, 1), cfg.kp_l0, device=self.device)
        self._kd_l0 = torch.full((self.num_envs, 1), cfg.kd_l0, device=self.device)
        self._torque_scale = torch.ones((self.num_envs, 6), device=self.device)

        if cfg.randomize_vmc_gains:
            self._randomize_gains()

        physics_dt = getattr(env, "physics_dt", env.step_dt / env.cfg.decimation)
        delay_min_ms, delay_max_ms = cfg.action_delay_ms_range
        self._action_delay_min_steps = max(0, int(round(delay_min_ms * 0.001 / physics_dt)))
        self._action_delay_max_steps = max(self._action_delay_min_steps, int(round(delay_max_ms * 0.001 / physics_dt)))
        fifo_len = self._action_delay_max_steps + 1
        self._action_fifo = torch.zeros(self.num_envs, fifo_len, self.action_dim, device=self.device)
        self._action_delay_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._randomize_action_delay()

    def _randomize_gains(self):
        """Randomize legacy gain buffers, wheel damping, and motor torque scale per environment.

        The leg PD buffers are kept for compatibility with older configs; direct
        task-space force control uses wheel damping and motor torque scale here.
        """
        low, high = self.cfg.gain_randomization_range
        self._kp_theta[:] = self.cfg.kp_theta * (low + (high - low) * torch.rand(self.num_envs, 1, device=self.device))
        self._kd_theta[:] = self.cfg.kd_theta * (low + (high - low) * torch.rand(self.num_envs, 1, device=self.device))
        self._kp_l0[:] = self.cfg.kp_l0 * (low + (high - low) * torch.rand(self.num_envs, 1, device=self.device))
        self._kd_l0[:] = self.cfg.kd_l0 * (low + (high - low) * torch.rand(self.num_envs, 1, device=self.device))
        self._torque_scale[:] = low + (high - low) * torch.rand(self.num_envs, 6, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def previous_actions(self) -> torch.Tensor:
        return self._previous_actions

    @property
    def previous_previous_actions(self) -> torch.Tensor:
        return self._previous_previous_actions

    def _randomize_action_delay(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            count = self.num_envs
        else:
            count = len(env_ids)
        if self.cfg.randomize_action_delay:
            self._action_delay_idx[env_ids] = torch.randint(
                self._action_delay_min_steps,
                self._action_delay_max_steps + 1,
                (count,),
                device=self.device,
            )
        else:
            self._action_delay_idx[env_ids] = 0

    def process_actions(self, actions: torch.Tensor):
        self._previous_previous_actions[:] = self._previous_actions
        self._previous_actions[:] = self._processed_actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions
        self._processed_actions[:, [0, 3]].clamp_(-self.cfg.clip_tp_actions, self.cfg.clip_tp_actions)
        self._processed_actions[:, [1, 4]].clamp_(-self.cfg.clip_force_actions, self.cfg.clip_force_actions)
        self._processed_actions[:, [2, 5]].clamp_(-self.cfg.clip_wheel_actions, self.cfg.clip_wheel_actions)

    def apply_actions(self):
        self._action_fifo[:, 1:, :] = self._action_fifo[:, :-1, :].clone()
        self._action_fifo[:, 0, :] = self._processed_actions
        self._delayed_actions[:] = self._action_fifo[torch.arange(self.num_envs, device=self.device), self._action_delay_idx]

        torques = compute_vmc_action(
            actions=self._delayed_actions,
            dof_pos=self._asset.data.joint_pos,
            dof_vel=self._asset.data.joint_vel,
            leg_joint_indices=self._leg_joint_ids,
            wheel_joint_indices=self._wheel_joint_ids,
            l1=self.cfg.l1,
            l2=self.cfg.l2,
            offset=self.cfg.offset,
            theta1_offset=self.cfg.theta1_offset,
            theta2_offset=self.cfg.theta2_offset,
            theta0_offset=self.cfg.theta0_offset,
            kp_theta=self._kp_theta,
            kd_theta=self._kd_theta,
            kp_l0=self._kp_l0,
            kd_l0=self._kd_l0,
            l0_offset=self.cfg.l0_offset,
            l0_min=self.cfg.l0_min,
            l0_max=self.cfg.l0_max,
            feedforward_force=self.cfg.feedforward_force,
            action_scale_tp=self.cfg.action_scale_tp,
            action_scale_force=self.cfg.action_scale_force,
            action_scale_wheel_torque=self.cfg.action_scale_wheel_torque,
            torque_limits=self._asset.data.soft_joint_pos_limits.new_tensor(self.cfg.torque_limits),
            torque_scale=self._torque_scale,
        )
        self._last_torques[:] = torques
        self._asset.set_joint_effort_target(torques)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._delayed_actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0
        self._action_fifo[env_ids] = 0.0
        self._randomize_action_delay(env_ids)


@configclass
class WLVMCActionCfg(ActionTermCfg):
    """Configuration for the WL VMC task-space force action term.

    Action order is [Tp_l, deltaF_l, wheel_l, Tp_r, deltaF_r, wheel_r].
    deltaF is added to feedforward_force before the VMC Jacobian mapping.
    """

    class_type: type[ActionTerm] = WLVMCAction

    leg_joint_names: list[str] = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
    wheel_joint_names: list[str] = ["l_wheel_Joint", "r_wheel_Joint"]
    l1: float = 0.21665632675675972
    l2: float = 0.2540023491164531
    offset: float = -0.007712217793726145
    theta1_offset: float = 0.14299916248023697
    theta2_offset: float = 2.406020345452543
    theta0_offset: float = 0.0
    # Legacy target-PD fields retained so older config code can still assign them.
    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 20.0
    l0_offset: float = 0.13
    l0_min: float = 0.1219258562330587
    l0_max: float = 0.3006386827708927
    feedforward_force: float = 40.0
    action_scale_tp: float = 15.0
    action_scale_force: float = 40.0
    action_scale_wheel_torque: float = 4.0
    clip_actions: float = 100.0
    clip_tp_actions: float = 1.0
    clip_force_actions: float = 1.0
    clip_wheel_actions: float = 3.0
    # Full articulation joint order is [lf0, rf0, lf1, rf1, l_wheel, r_wheel].
    torque_limits: list[float] = [50.0, 50.0, 50.0, 50.0, 4.0, 4.0]

    # Domain randomization
    randomize_vmc_gains: bool = False
    gain_randomization_range: tuple[float, float] = (0.9, 1.1)
    randomize_action_delay: bool = True
    action_delay_ms_range: tuple[float, float] = (0.0, 10.0)


