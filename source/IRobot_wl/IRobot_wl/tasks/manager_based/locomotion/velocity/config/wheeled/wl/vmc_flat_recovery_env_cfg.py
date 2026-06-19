"""VMC Flat terrain environment with stand-up recovery training.

This module extends the standard VMC flat environment with:
  - Random fallen initial poses for stand-up recovery learning
  - Recovery-specific reward terms blended with locomotion rewards
  - Relaxed termination conditions during recovery phase
  - Warm-start compatible: policy obs dim unchanged (27), recovery_mode in critic only
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp
from .vmc_flat_env_cfg import WLVMCVanillaFlatEnvCfg


# ============================================================================ #
# Recovery Environment Configuration
# ============================================================================ #


@configclass
class WLVMCVanillaFlatRecoveryEnvCfg(WLVMCVanillaFlatEnvCfg):
    """VMC flat terrain environment with stand-up recovery training.

    Extends :class:`WLVMCVanillaFlatEnvCfg` with:
      - Random fallen initial poses for recovery learning
      - Recovery-specific reward terms (merged into self.rewards)
      - Relaxed termination (no illegal_contact termination)
      - Extended episode length for recovery + locomotion
      - Wider domain randomization

    .. note::
        Policy observation dimension is kept at 27 (same as standard flat)
        to enable direct warm-start from existing locomotion checkpoints.
        The ``projected_gravity`` in policy obs already signals orientation.
        A ``recovery_mode`` flag is added to critic obs as privileged info.
    """

    # Fraction of environments that start in a fallen pose each reset.
    # Set to 1.0 for full recovery training, or lower for curriculum mixing.
    fallen_probability: float = 0.5

    def __post_init__(self):
        # Post init of parent chain:
        #   WLVMCVanillaFlatEnvCfg → WLVMCVanillaRoughEnvCfg → LocomotionVelocityRoughEnvCfg
        super().__post_init__()

        # Longer episode: time to recover (~15s) + normal locomotion (~60s)
        self.episode_length_s = 75.0

        # ------------------------------Observations------------------------------
        # Add recovery_mode to critic as privileged information.
        # Policy obs stays at 27 dims — projected_gravity already encodes orientation.
        self.observations.critic.recovery_mode = ObsTerm(
            func=mdp.recovery_mode_obs,
            params={"upright_threshold": -0.7, "fallen_threshold": -0.3},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # ------------------------------Events------------------------------
        # Replace base reset with fallen-aware reset
        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_root_state_fallen,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "z": (0.0, 0.0),
                    "roll": (-3.14, 3.14),
                    "pitch": (-3.14, 3.14),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
                "joint_position_range": (-0.5, 0.5),
                "joint_velocity_range": (-1.0, 1.0),
                "fallen_probability": self.fallen_probability,
                "ground_height_offset": 0.05,
            },
        )

        # Keep joint noise for all envs (fallen reset handles joint positions internally)
        self.events.randomize_reset_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
            },
        )

        # Stronger pushes to trigger recovery during locomotion
        self.events.randomize_push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
        )

        # Wider friction range: recovery needs grip to push against ground
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.3, 2.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.3, 1.5)

        # ------------------------------Terminations------------------------------
        # Disable illegal_contact: body/leg contact with ground is expected during recovery.
        # The 75s episode time_out provides an upper bound.
        self.terminations.illegal_contact = None

        # ------------------------------Rewards------------------------------
        # Merge recovery-specific reward terms into self.rewards so that
        # Isaac Lab's RewardManager discovers and processes them automatically.
        #
        # These rewards are always active. The upright_factor in locomotion
        # rewards naturally scales them to ~0 when fallen, letting recovery
        # rewards dominate. As the robot stands up and upright_factor → 1,
        # locomotion rewards regain their full weight.

        # Core stand-up progress: exponential reward for projected_gravity_z → -1
        self.rewards.recovery_upright_progress = RewTerm(
            func=mdp.recovery_upright_progress,
            weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Reward lifting base off the ground toward target height
        self.rewards.recovery_base_height = RewTerm(
            func=mdp.recovery_base_height,
            weight=2.0,
            params={
                "target_height": 0.23,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Reward extending legs to push body up
        self.rewards.recovery_leg_extension = RewTerm(
            func=mdp.recovery_leg_extension,
            weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "leg_joint_names": ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                "l1": 0.21665632675675972,
                "l2": 0.2540023491164531,
                "offset": -0.007712217793726145,
                "theta1_offset": 0.14299916248023697,
                "theta2_offset": 2.406020345452543,
                "l0_min": 0.1219258562330587,
                "l0_max": 0.3006386827708927,
            },
        )

        # Penalize left/right asymmetry during recovery
        self.rewards.recovery_leg_symmetry = RewTerm(
            func=mdp.recovery_leg_symmetry,
            weight=-1.5,
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

        # Penalize jerky actions during stand-up
        self.rewards.recovery_action_smoothness = RewTerm(
            func=mdp.recovery_action_smoothness,
            weight=-0.003,
        )

        # Encourage active motion (wheel assist) when fallen
        self.rewards.recovery_wheel_assist = RewTerm(
            func=mdp.recovery_wheel_assist,
            weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ------------------------------Commands------------------------------
        # Keep existing command distribution from parent (10% stop, 30% straight, 30% turn, 30% arc).
        # When fallen, upright_factor scales tracking rewards to ~0, so velocity
        # commands are effectively ignored. Policy learns this from projected_gravity.

        # ------------------------------Curriculum------------------------------
        # Inherited from WLVMCVanillaFlatEnvCfg

        # Clean up zero-weight rewards
        if self.__class__.__name__ == "WLVMCVanillaFlatRecoveryEnvCfg":
            self.disable_zero_weight_rewards()
