"""Configuration for the simple wheel-legged robot (WL)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from IRobot_wl.assets import IROBOT_WL_DATA_DIR


WL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{IROBOT_WL_DATA_DIR}/Robots/wl/wl_description/urdf/wl.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.24019077554782198),
        joint_pos={
            "lf0_Joint": 0.043909646385,
            "lf1_Joint": -0.081805500076,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.043909646385,
            "rf1_Joint": 0.081805500076,
            "r_wheel_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=["lf0_Joint", "rf0_Joint"],
            effort_limit=30.0,
            saturation_effort=30.0,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=["lf1_Joint", "rf1_Joint"],
            effort_limit=30.0,
            saturation_effort=30.0,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=["l_wheel_Joint", "r_wheel_Joint"],
            effort_limit=4.0,
            saturation_effort=4.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
)
