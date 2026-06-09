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
    dt: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Compute the mirrored leg-frame state used by the original WL-Gym VMC."""

    theta1 = torch.stack(
        [dof_pos[:, leg_joint_indices[0]], -dof_pos[:, leg_joint_indices[2]]], dim=1
    )
    theta2 = torch.stack(
        [dof_pos[:, leg_joint_indices[1]] + torch.pi / 2, -dof_pos[:, leg_joint_indices[3]] + torch.pi / 2], dim=1
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
    theta0_dot = (theta0_fwd - theta0) / dt

    wheel_pos = torch.stack(
        [dof_pos[:, wheel_joint_indices[0]], dof_pos[:, wheel_joint_indices[1]]], dim=1
    )
    wheel_vel = torch.stack(
        [dof_vel[:, wheel_joint_indices[0]], dof_vel[:, wheel_joint_indices[1]]], dim=1
    )

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
    # Clamp L0 to avoid numerical issues with acos
    L0_clamped = torch.clamp(L0, min=abs(l1 - l2) + 1e-4, max=l1 + l2 - 1e-4)

    # Law of cosines for the knee angle
    cos_beta = (l1**2 + l2**2 - L0_clamped**2) / (2.0 * l1 * l2)
    cos_beta = torch.clamp(cos_beta, -1.0 + 1e-6, 1.0 - 1e-6)
    beta = torch.acos(cos_beta)  # angle between thigh and calf, 0 = fully folded

    # Knee joint angle: theta2 = pi - beta (0 = straight leg, positive = backward bend)
    theta2 = torch.pi - beta

    # Hip joint angle from geometry
    # alpha = angle between thigh and the line from hip to end-effector
    gamma = theta0 + torch.pi / 2  # convert to coordinate frame
    alpha = torch.atan2(l2 * torch.sin(theta2), l1 + l2 * torch.cos(theta2))
    theta1 = gamma - alpha

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
    T_leg (about hip, positive = counterclockwise), compute the corresponding joint
    torques for hip (T1) and knee (T2).

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
    theta0_shifted = theta0 + torch.pi / 2

    # Jacobian transpose elements for the 2-DOF leg
    t11 = l1 * torch.sin(theta0_shifted - theta1) - l2 * torch.sin(theta1 + theta2 - theta0_shifted)
    t12 = (l1 * torch.cos(theta0_shifted - theta1) - l2 * torch.cos(theta1 + theta2 - theta0_shifted)) / L0

    t21 = -l2 * torch.sin(theta1 + theta2 - theta0_shifted)
    t22 = -l2 * torch.cos(theta1 + theta2 - theta0_shifted) / L0

    T1 = t11 * F_leg - t12 * T_leg
    T2 = t21 * F_leg - t22 * T_leg

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
    kp_theta: float,
    kd_theta: float,
    kp_l0: float,
    kd_l0: float,
    l0_offset: float,
    feedforward_force: float,
    action_scale_theta: float,
    action_scale_l0: float,
    action_scale_vel: float,
    wheel_damping: float,
    torque_limits: torch.Tensor,
) -> torch.Tensor:
    """Compute joint torques from VMC task-space actions.

    This function takes policy actions in task-space coordinates and converts them
    to joint torques using Virtual Model Control. The action space is 6-dimensional
    for a bipedal robot with 2 legs:
      [theta0_left, L0_left, wheel_vel_left, theta0_right, L0_right, wheel_vel_right]

    Args:
        actions: Policy actions in task space, shape (num_envs, 6).
        dof_pos: Current joint positions, shape (num_envs, num_dof).
        dof_vel: Current joint velocities, shape (num_envs, num_dof).
        leg_joint_indices: Indices of leg joints [hip_left, knee_left, hip_right, knee_right].
        wheel_joint_indices: Indices of wheel joints [wheel_left, wheel_right].
        l1: Thigh link length [m].
        l2: Calf link length [m].
        offset: Hip offset from body center [m].
        kp_theta: Angle PD proportional gain [Nm/rad].
        kd_theta: Angle PD derivative gain [Nm*s/rad].
        kp_l0: Length PD proportional gain [N/m].
        kd_l0: Length PD derivative gain [N*s/m].
        l0_offset: Default leg length offset [m].
        feedforward_force: Gravity compensation force [N].
        action_scale_theta: Scale for theta actions.
        action_scale_l0: Scale for L0 actions.
        action_scale_vel: Scale for wheel velocity actions.
        wheel_damping: Damping for wheel velocity control [Nm*s/rad].
        torque_limits: Joint torque limits, shape (num_dof,).

    Returns:
        torques: Joint torques, shape (num_envs, num_dof).
    """
    num_envs = actions.shape[0]
    num_dof = dof_pos.shape[1]

    # --- Parse task-space actions ---
    # Left leg: actions[:, 0:3], Right leg: actions[:, 3:6]
    theta0_ref = torch.stack([actions[:, 0], actions[:, 3]], dim=1)  # (num_envs, 2)
    theta0_ref = theta0_ref * action_scale_theta

    l0_ref = torch.stack([actions[:, 1], actions[:, 4]], dim=1)  # (num_envs, 2)
    l0_ref = l0_ref * action_scale_l0 + l0_offset

    wheel_vel_ref = torch.stack([actions[:, 2], actions[:, 5]], dim=1)  # (num_envs, 2)
    wheel_vel_ref = wheel_vel_ref * action_scale_vel

    state = compute_vmc_state(
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        leg_joint_indices=leg_joint_indices,
        wheel_joint_indices=wheel_joint_indices,
        l1=l1,
        l2=l2,
        offset=offset,
    )
    theta1 = state["theta1"]
    theta2 = state["theta2"]
    L0 = state["L0"]
    theta0 = state["theta0"]
    L0_dot = state["L0_dot"]
    theta0_dot = state["theta0_dot"]
    wheel_vel = state["wheel_vel"]

    # --- Task-space PD control ---
    torque_leg = kp_theta * (theta0_ref - theta0) - kd_theta * theta0_dot
    force_leg = kp_l0 * (l0_ref - L0) - kd_l0 * L0_dot

    # --- VMC: task-space force/torque → joint torques ---
    T1, T2 = vmc_torques(theta0, theta1, theta2, L0, force_leg + feedforward_force, torque_leg, l1, l2)

    # --- Wheel velocity PD control ---
    torque_wheel = wheel_damping * (wheel_vel_ref - wheel_vel)

    # --- Assemble full torque vector ---
    torques = torch.zeros(num_envs, num_dof, device=actions.device, dtype=actions.dtype)

    # Left leg
    torques[:, leg_joint_indices[0]] = T1[:, 0]  # left hip
    torques[:, leg_joint_indices[1]] = T2[:, 0]  # left knee

    # Right leg (negate due to mirror convention)
    torques[:, leg_joint_indices[2]] = -T1[:, 1]  # right hip
    torques[:, leg_joint_indices[3]] = -T2[:, 1]  # right knee

    # Wheels
    torques[:, wheel_joint_indices[0]] = torque_wheel[:, 0]  # left wheel
    torques[:, wheel_joint_indices[1]] = torque_wheel[:, 1]  # right wheel

    # Clip to torque limits
    torques = torch.clamp(torques, -torque_limits, torque_limits)

    return torques


class WLVMCAction(ActionTerm):
    """Direct torque VMC action term matching the original Wheel-Legged-Gym semantics."""

    cfg: "WLVMCActionCfg"

    def __init__(self, cfg: "WLVMCActionCfg", env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._leg_joint_ids, _ = self._asset.find_joints(cfg.leg_joint_names, preserve_order=True)
        self._wheel_joint_ids, _ = self._asset.find_joints(cfg.wheel_joint_names, preserve_order=True)
    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -self.cfg.clip_actions, self.cfg.clip_actions)
        self._processed_actions[:] = self._raw_actions

    def apply_actions(self):
        torques = compute_vmc_action(
            actions=self._processed_actions,
            dof_pos=self._asset.data.joint_pos,
            dof_vel=self._asset.data.joint_vel,
            leg_joint_indices=self._leg_joint_ids,
            wheel_joint_indices=self._wheel_joint_ids,
            l1=self.cfg.l1,
            l2=self.cfg.l2,
            offset=self.cfg.offset,
            kp_theta=self.cfg.kp_theta,
            kd_theta=self.cfg.kd_theta,
            kp_l0=self.cfg.kp_l0,
            kd_l0=self.cfg.kd_l0,
            l0_offset=self.cfg.l0_offset,
            feedforward_force=self.cfg.feedforward_force,
            action_scale_theta=self.cfg.action_scale_theta,
            action_scale_l0=self.cfg.action_scale_l0,
            action_scale_vel=self.cfg.action_scale_vel,
            wheel_damping=self.cfg.wheel_damping,
            torque_limits=self._asset.data.soft_joint_pos_limits.new_tensor(self.cfg.torque_limits),
        )
        self._asset.set_joint_effort_target(torques)

    def reset(self, env_ids=None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class WLVMCActionCfg(ActionTermCfg):
    """Configuration for the direct-torque WL VMC action term."""

    class_type: type[ActionTerm] = WLVMCAction

    leg_joint_names: list[str] = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
    wheel_joint_names: list[str] = ["l_wheel_Joint", "r_wheel_Joint"]
    l1: float = 0.15
    l2: float = 0.25
    offset: float = 0.054
    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 20.0
    l0_offset: float = 0.19
    feedforward_force: float = 40.0
    action_scale_theta: float = 0.5
    action_scale_l0: float = 0.1
    action_scale_vel: float = 10.0
    wheel_damping: float = 0.5
    clip_actions: float = 100.0
    torque_limits: list[float] = [30.0, 30.0, 30.0, 30.0, 5.0, 5.0]


def convert_vmc_to_joint_actions(
    actions: torch.Tensor,
    l1: float,
    l2: float,
    offset: float,
    l0_offset: float,
    action_scale_theta: float,
    action_scale_l0: float,
    action_scale_vel: float,
) -> torch.Tensor:
    """Convert task-space actions to joint-space targets for Isaac Lab's PD pipeline.

    The policy outputs 6-dim task-space actions:
      [theta0_l, L0_l, wheel_vel_l, theta0_r, L0_r, wheel_vel_r]

    This function uses IK to convert (theta0, L0) → (theta1, theta2) for each leg,
    producing 6-dim joint-space targets:
      [hip_l, knee_l, hip_r, knee_r, wheel_vel_l, wheel_vel_r]

    The output is designed to be fed directly to Isaac Lab's JointPositionAction
    (leg joints) + JointVelocityAction (wheel joints).

    Args:
        actions: Policy actions in task space, shape (num_envs, 6).
        l1: Thigh link length [m].
        l2: Calf link length [m].
        offset: Hip offset from body center [m].
        l0_offset: Default leg length offset [m].
        action_scale_theta: Scale for theta actions.
        action_scale_l0: Scale for L0 actions.
        action_scale_vel: Scale for wheel velocity actions.

    Returns:
        Joint-space action vector, shape (num_envs, 6).
        Order: [hip_l, knee_l, hip_r, knee_r, wheel_vel_l, wheel_vel_r]
    """
    # Parse task-space actions: [theta0_l, L0_l, w_l, theta0_r, L0_r, w_r]
    theta0 = torch.stack([actions[:, 0], actions[:, 3]], dim=1)  # (num_envs, 2)
    theta0 = theta0 * action_scale_theta

    L0 = torch.stack([actions[:, 1], actions[:, 4]], dim=1)  # (num_envs, 2)
    L0 = L0 * action_scale_l0 + l0_offset

    wheel_vel = torch.stack([actions[:, 2], actions[:, 5]], dim=1)  # (num_envs, 2)
    wheel_vel = wheel_vel * action_scale_vel

    # IK: task-space → mirrored leg-frame joint angles used by the original
    # Wheel-Legged-Gym VMC implementation.
    theta1, theta2 = inverse_kinematics(theta0, L0, l1, l2, offset)

    # Map leg-frame angles back to the physical joint coordinates expected by
    # the URDF / Isaac Lab articulation. In the original Isaac Gym task:
    #   left  leg: theta1 =  q_lf0, theta2 =  q_lf1 + pi/2
    #   right leg: theta1 = -q_rf0, theta2 = -q_rf1 + pi/2
    # Therefore the inverse mapping is:
    #   q_lf0 =  theta1_l
    #   q_lf1 =  theta2_l - pi/2
    #   q_rf0 = -theta1_r
    #   q_rf1 =  pi/2 - theta2_r
    hip_l = theta1[:, 0:1]
    knee_l = theta2[:, 0:1] - torch.pi / 2
    hip_r = -theta1[:, 1:2]
    knee_r = torch.pi / 2 - theta2[:, 1:2]

    # Assemble: [hip_l, knee_l, hip_r, knee_r, wheel_vel_l, wheel_vel_r]
    joint_actions = torch.cat(
        [hip_l, knee_l, hip_r, knee_r, wheel_vel[:, 0:1], wheel_vel[:, 1:2]],
        dim=1,
    )
    return joint_actions


def apply_vmc_wrapper(env):
    """Wrap an Isaac Lab environment to convert VMC task-space actions to joint-space.

    This monkey-patches the env to intercept policy actions and convert them
    from task-space [theta0, L0, wheel_vel] × 2 legs to joint-space
    [hip, knee] × 2 legs + [wheel_vel] × 2 wheels via inverse kinematics.

    Call this after gym.make() and before training:

        env = gym.make("IRobot-WL-Velocity-VMC-Flat-v0", cfg=env_cfg)
        env = apply_vmc_wrapper(env)
        env = RslRlVecEnvWrapper(env, ...)

    Args:
        env: The Isaac Lab ManagerBasedRLEnv instance (possibly wrapped by gym wrappers).

    Returns:
        The wrapped environment (same object, with patched step method).
    """
    # Handle gym wrappers (OrderEnforcing, RecordVideo, etc.) by unwrapping
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    vmc_cfg = base_env.cfg.vmc_actions
    original_step = env.step

    def vmc_step(actions):
        joint_actions = convert_vmc_to_joint_actions(
            actions=actions,
            l1=vmc_cfg.l1,
            l2=vmc_cfg.l2,
            offset=vmc_cfg.offset,
            l0_offset=vmc_cfg.l0_offset,
            action_scale_theta=vmc_cfg.action_scale_theta,
            action_scale_l0=vmc_cfg.action_scale_l0,
            action_scale_vel=vmc_cfg.action_scale_vel,
        )
        return original_step(joint_actions)

    env.step = vmc_step
    return env
