"""VMC (Virtual Model Control) environment configuration for the WL robot.

This module defines the VMC-based training environment for the simple 2-DOF leg
wheel-legged robot. The policy works in task space (leg angle θ₀, leg length L₀,
wheel velocity) and VMC converts these to joint torques.

Reference: Wheel-Legged-Gym wheel_legged_vmc task.
"""

import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp
from IRobot_wl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from IRobot_wl.assets.wl import WL_CFG  # isort: skip
from .rough_env_cfg import ROUGH_ROAD_CFG  # isort: skip


# ============================================================================ #
# VMC Action Configuration
# ============================================================================ #


@configclass
class WLVMCVanillaActionsCfg:
    """Action config that stores VMC task-space action parameters.

    Unlike standard Isaac Lab action configs, this stores VMC parameters
    directly on the config object. The actual torque computation happens
    in the step loop via compute_vmc_torques_from_actions().
    """

    # Number of actions: 6 (theta0, L0, wheel_vel per leg × 2 legs)
    num_actions = 6

    # VMC geometry parameters (from URDF)
    l1: float = 0.21665632675675972  # thigh length [m]
    l2: float = 0.2540023491164531  # calf length [m]
    offset: float = -0.007712217793726145  # hip offset [m]
    theta1_offset: float = 0.14299916248023697  # first link zero-angle offset [rad]
    theta2_offset: float = 2.406020345452543  # knee-to-wheel zero-angle offset [rad]
    theta0_offset: float = 0.0  # default task-space leg angle [rad]

    # VMC PD gains
    kp_theta: float = 50.0  # angle P gain [Nm/rad]
    kd_theta: float = 3.0  # angle D gain [Nm*s/rad]
    kp_l0: float = 900.0  # length P gain [N/m]
    kd_l0: float = 20.0  # length D gain [N*s/m]

    # VMC parameters
    l0_offset: float = 0.19  # default leg length [m]
    l0_min: float = 0.1219258562330587  # reachable L0 lower bound [m]
    l0_max: float = 0.3006386827708927  # reachable L0 upper bound [m]
    feedforward_force: float = 40.0  # gravity compensation [N]

    # Action scales
    action_scale_theta: float = 0.5
    action_scale_l0: float = 0.1
    action_scale_vel: float = 10.0

    # Wheel control
    wheel_damping: float = 0.05  # damping for wheel velocity PD [Nm*s/rad]

    # Action clipping
    clip_actions: float = 100.0


# ============================================================================ #
# VMC Observation Configuration
# ============================================================================ #


@configclass
class WLVMCVanillaObservationsCfg:
    """VMC task-space observations configuration.

    In VMC mode, we observe:
      - Task-space state: theta0, L0, theta0_dot, L0_dot (2 legs × 4 = 8 dims)
      - Wheel state: wheel_pos, wheel_vel (2 wheels × 2 = 4 dims)
      - Base state: ang_vel (3), projected_gravity (3)
      - Commands (3)
      - Previous actions (6)
      Total: 8 + 4 + 3 + 3 + 3 + 6 = 27 dims
    """

    # Base observations
    base_ang_vel = True  # 3
    projected_gravity = True  # 3

    # Task-space observations
    leg_angle = True  # theta0, 2
    leg_length = True  # L0, 2
    leg_angle_dot = True  # theta0_dot, 2
    leg_length_dot = True  # L0_dot, 2

    # Wheel observations
    wheel_pos = True  # 2
    wheel_vel = True  # 2

    # Commands
    velocity_commands = True  # 3

    # Action history
    last_action = True  # 6


@configclass
class WLVMCControlActionsCfg:
    """Direct-torque VMC action specification."""

    vmc = mdp.WLVMCActionCfg(
        asset_name="robot",
        leg_joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
        wheel_joint_names=["l_wheel_Joint", "r_wheel_Joint"],
        l1=0.21665632675675972,
        l2=0.2540023491164531,
        offset=-0.007712217793726145,
        theta1_offset=0.14299916248023697,
        theta2_offset=2.406020345452543,
        theta0_offset=0.0,
        kp_theta=50.0,
        kd_theta=3.0,
        kp_l0=900.0,
        kd_l0=20.0,
        l0_offset=0.19,
        l0_min=0.1219258562330587,
        l0_max=0.3006386827708927,
        feedforward_force=40.0,
        action_scale_theta=0.5,
        action_scale_l0=0.1,
        action_scale_vel=10.0,
        wheel_damping=0.05,
        clip_actions=100.0,
        # Full articulation joint order is [lf0, rf0, lf1, rf1, l_wheel, r_wheel].
        torque_limits=[30.0, 30.0, 30.0, 30.0, 4.0, 4.0],
        randomize_action_delay=True,
        action_delay_ms_range=(0.0, 10.0),
    )


@configclass
class WLVMCObsCfg(ObservationsCfg):
    """Observation layout matched to the old Wheel-Legged-Gym VMC task."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.wl_vmc_commands,
            params={"command_name": "base_velocity", "height_command": 0.25},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        leg_angle = ObsTerm(
            func=mdp.leg_angle,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        leg_angle_dot = ObsTerm(
            func=mdp.leg_angle_dot,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "wheel_joint_names": ["l_wheel_Joint", "r_wheel_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        leg_length = ObsTerm(
            func=mdp.leg_length,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-100.0, 100.0),
            scale=5.0,
        )
        leg_length_dot = ObsTerm(
            func=mdp.leg_length_dot,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "wheel_joint_names": ["l_wheel_Joint", "r_wheel_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        wheel_pos = ObsTerm(
            func=mdp.wheel_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_Joint", "r_wheel_Joint"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        wheel_vel = ObsTerm(
            func=mdp.wheel_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_Joint", "r_wheel_Joint"])},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "vmc"}, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyHistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.wl_vmc_commands,
            params={"command_name": "base_velocity", "height_command": 0.25},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        leg_angle = ObsTerm(
            func=mdp.leg_angle,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        leg_angle_dot = ObsTerm(
            func=mdp.leg_angle_dot,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "wheel_joint_names": ["l_wheel_Joint", "r_wheel_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        leg_length = ObsTerm(
            func=mdp.leg_length,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            clip=(-100.0, 100.0),
            scale=5.0,
        )
        leg_length_dot = ObsTerm(
            func=mdp.leg_length_dot,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "wheel_joint_names": ["l_wheel_Joint", "r_wheel_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
            },
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        wheel_pos = ObsTerm(
            func=mdp.wheel_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_Joint", "r_wheel_Joint"])},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        wheel_vel = ObsTerm(
            func=mdp.wheel_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_Joint", "r_wheel_Joint"])},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "vmc"}, clip=(-100.0, 100.0), scale=1.0)
        prev_actions = ObsTerm(func=mdp.previous_action, params={"action_name": "vmc"}, clip=(-100.0, 100.0), scale=1.0)
        prev_prev_actions = ObsTerm(
            func=mdp.previous_previous_action,
            params={"action_name": "vmc"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint", "l_wheel_Joint", "r_wheel_Joint"],
                )
            },
            clip=(-100.0, 100.0),
            scale=0.0025,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint", "l_wheel_Joint", "r_wheel_Joint"],
                ),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_Joint", "r_wheel_Joint"]),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint", "l_wheel_Joint", "r_wheel_Joint"],
                )
            },
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-100.0, 100.0),
            scale=5.0,
        )
        applied_torque = ObsTerm(
            func=mdp.applied_joint_torque,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint", "l_wheel_Joint", "r_wheel_Joint"],
                )
            },
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        base_mass_delta = ObsTerm(
            func=mdp.body_mass_delta,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_com = ObsTerm(
            func=mdp.body_com_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        default_joint_pos_delta = ObsTerm(
            func=mdp.default_joint_pos_delta,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint", "l_wheel_Joint", "r_wheel_Joint"],
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        friction = ObsTerm(
            func=mdp.material_static_friction,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        restitution = ObsTerm(
            func=mdp.material_restitution,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    policy_history: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================================ #
# VMC Environment Configuration
# ============================================================================ #


@configclass
class WLVMCVanillaRewardsCfg(RewardsCfg):
    """Reward terms for the VMC MDP, matching WL-Gym's reward structure."""

    # VMC-specific rewards (added to base)
    nominal_state = RewTerm(
        func=mdp.nominal_state,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
            "l1": 0.21665632675675972,
            "l2": 0.2540023491164531,
            "offset": -0.007712217793726145,
            "theta1_offset": 0.14299916248023697,
            "theta2_offset": 2.406020345452543,
        },
    )
    leg_length_symmetry = RewTerm(
        func=mdp.leg_length_symmetry,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
            "l1": 0.21665632675675972,
            "l2": 0.2540023491164531,
            "offset": -0.007712217793726145,
            "theta1_offset": 0.14299916248023697,
            "theta2_offset": 2.406020345452543,
        },
    )
    vmc_action_symmetry = RewTerm(
        func=mdp.vmc_action_symmetry,
        weight=-0.25,
        params={"action_name": "vmc"},
    )
    base_height_enhance = RewTerm(
        func=mdp.base_height_enhance,
        weight=0.0,
        params={"target_height": 0.25, "sensor_cfg": None},
    )

    # Wheel-specific rewards (added to base)
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class WLVMCVanillaRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """VMC-based wheel-legged locomotion environment on rough terrain."""

    observations: WLVMCObsCfg = WLVMCObsCfg()
    actions: WLVMCControlActionsCfg = WLVMCControlActionsCfg()
    rewards: WLVMCVanillaRewardsCfg = WLVMCVanillaRewardsCfg()

    # VMC action config (stored for reference; actual action processing
    # is handled by overriding the action pipeline in __post_init__)
    vmc_actions: WLVMCVanillaActionsCfg = WLVMCVanillaActionsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*wheel_Link"

    # fmt: off
    leg_joint_names = [
        "lf0_Joint", "lf1_Joint",
        "rf0_Joint", "rf1_Joint",
    ]
    wheel_joint_names = [
        "l_wheel_Joint", "r_wheel_Joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # spread out environments for better visualization
        self.scene.env_spacing = 4.0

        # ------------------------------Scene------------------------------
        self.scene.robot = WL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_ROAD_CFG

        # ------------------------------Observations------------------------------
        self.observations.policy.velocity_commands.params["height_command"] = self.rewards.base_height_enhance.params["target_height"]
        self.observations.critic.velocity_commands.params["height_command"] = self.rewards.base_height_enhance.params["target_height"]

        # ------------------------------Actions------------------------------
        self.actions.vmc.leg_joint_names = self.leg_joint_names
        self.actions.vmc.wheel_joint_names = self.wheel_joint_names
        self.actions.vmc.l1 = self.vmc_actions.l1
        self.actions.vmc.l2 = self.vmc_actions.l2
        self.actions.vmc.offset = self.vmc_actions.offset
        self.actions.vmc.theta1_offset = self.vmc_actions.theta1_offset
        self.actions.vmc.theta2_offset = self.vmc_actions.theta2_offset
        self.actions.vmc.theta0_offset = self.vmc_actions.theta0_offset
        self.actions.vmc.kp_theta = self.vmc_actions.kp_theta
        self.actions.vmc.kd_theta = self.vmc_actions.kd_theta
        self.actions.vmc.kp_l0 = self.vmc_actions.kp_l0
        self.actions.vmc.kd_l0 = self.vmc_actions.kd_l0
        self.actions.vmc.l0_offset = self.vmc_actions.l0_offset
        self.actions.vmc.l0_min = self.vmc_actions.l0_min
        self.actions.vmc.l0_max = self.vmc_actions.l0_max
        self.actions.vmc.feedforward_force = self.vmc_actions.feedforward_force
        self.actions.vmc.action_scale_theta = self.vmc_actions.action_scale_theta
        self.actions.vmc.action_scale_l0 = self.vmc_actions.action_scale_l0
        self.actions.vmc.action_scale_vel = self.vmc_actions.action_scale_vel
        self.actions.vmc.wheel_damping = self.vmc_actions.wheel_damping
        self.actions.vmc.clip_actions = self.vmc_actions.clip_actions
        self.actions.vmc.randomize_action_delay = True
        self.actions.vmc.action_delay_ms_range = (0.0, 10.0)

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200

        # Root penalties (WL-Gym style)
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -10.0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.25
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_enhance.weight = 1.0
        self.rewards.base_height_enhance.params["target_height"] = 0.25
        self.rewards.nominal_state.weight = -0.3
        self.rewards.body_lin_acc_l2.weight = 0

        # Joint penalties (legs only for VMC)
        self.rewards.joint_torques_l2.weight = -1.0e-4
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = -5.0e-5
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = ["lf1_Joint", "rf1_Joint"]
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.params["velocity_threshold"] = 100
        self.rewards.wheel_vel_penalty.weight = -0.01
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["rf0_Joint", "lf0_Joint"],
            ["rf1_Joint", "lf1_Joint"],
        ]
        self.rewards.nominal_state.weight = -0.3
        self.rewards.leg_length_symmetry.weight = -4.0
        self.rewards.vmc_action_symmetry.weight = -0.25

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.action_smooth.weight = 0  # disabled, not in WLVMCVanillaRewardsCfg

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards (WL-Gym style with std=0.25)
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.tracking_lin_vel_enhance.weight = 1.0
        self.rewards.tracking_ang_vel_enhance.weight = 1.0

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("l_wheel_Joint", "r_wheel_Joint"),)
        self.rewards.upward.weight = 1.0
        self.rewards.feet_distance_y_exp.weight = 1.5
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.34
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WLVMCVanillaRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            ".*f0_Link",
            ".*f1_Link",
        ]

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel.params["reward_term_name"] = "track_lin_vel_xy_exp"
        self.curriculum.command_levels_lin_vel.params["threshold"] = 0.7
        self.curriculum.command_levels_ang_vel.params["reward_term_name"] = "track_ang_vel_z_exp"
        self.curriculum.command_levels_ang_vel.params["threshold"] = 0.7

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-3.14, 3.14)
        self.commands.base_velocity.heading_command = True


def compute_vmc_torques_from_actions(env, actions: torch.Tensor) -> torch.Tensor:
    """Convert VMC task-space actions to joint torques.

    This is the bridge between the VMC config and the Isaac Lab action pipeline.
    Call this function in the environment's step loop to convert policy actions
    to joint torques.

    Args:
        env: The ManagerBasedRLEnv instance (must have vmc_actions config).
        actions: Policy actions in task space, shape (num_envs, 6).
                 Order: [theta0_l, L0_l, w_l, theta0_r, L0_r, w_r]

    Returns:
        Joint torques for all 6 DOFs, shape (num_envs, 6).
    """
    vmc_cfg = env.cfg.vmc_actions
    robot = env.scene["robot"]
    leg_joint_indices, _ = robot.find_joints(env.cfg.leg_joint_names, preserve_order=True)
    wheel_joint_indices, _ = robot.find_joints(env.cfg.wheel_joint_names, preserve_order=True)

    return compute_vmc_action(
        actions=actions,
        dof_pos=robot.data.joint_pos,
        dof_vel=robot.data.joint_vel,
        leg_joint_indices=leg_joint_indices,
        wheel_joint_indices=wheel_joint_indices,
        l1=vmc_cfg.l1,
        l2=vmc_cfg.l2,
        offset=vmc_cfg.offset,
        theta1_offset=vmc_cfg.theta1_offset,
        theta2_offset=vmc_cfg.theta2_offset,
        theta0_offset=vmc_cfg.theta0_offset,
        kp_theta=vmc_cfg.kp_theta,
        kd_theta=vmc_cfg.kd_theta,
        kp_l0=vmc_cfg.kp_l0,
        kd_l0=vmc_cfg.kd_l0,
        l0_offset=vmc_cfg.l0_offset,
        l0_min=vmc_cfg.l0_min,
        l0_max=vmc_cfg.l0_max,
        feedforward_force=vmc_cfg.feedforward_force,
        action_scale_theta=vmc_cfg.action_scale_theta,
        action_scale_l0=vmc_cfg.action_scale_l0,
        action_scale_vel=vmc_cfg.action_scale_vel,
        wheel_damping=vmc_cfg.wheel_damping,
        torque_limits=robot.data.torque_limit,
    )
