from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize body inertia tensors with a positive uniform scale per body.

    The full 3x3 inertia tensor is scaled by one positive factor per
    environment/body. Scaling the complete tensor preserves positive-definiteness
    when the source inertia tensor is valid, unlike independently scaling only
    ``ixx``, ``iyy``, and ``izz`` while leaving off-diagonal terms unchanged.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    if operation != "scale":
        raise NotImplementedError("Safe inertia randomization only supports operation='scale'.")
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for inertia randomization."
            " Please use 'uniform', 'log_uniform', or 'gaussian'."
        )
    scale = dist_fn(*inertia_distribution_params, (len(env_ids), len(body_ids), 1), device=inertias.device)
    inertias[env_ids[:, None], body_ids, :] *= scale

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization,
            or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Note: If "pits" terrain exists, environments on pit terrain will be reset to default state without random
    perturbations to avoid the robot falling into the pit.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Separate pit and non-pit environments
    # Check which environments are assigned to pit terrain (not random reset)
    assigned_to_pits = is_env_assigned_to_terrain(env, "pits")
    pit_env_ids = env_ids[assigned_to_pits[env_ids]]
    non_pit_env_ids = env_ids[~assigned_to_pits[env_ids]]

    # Reset pit environments to default state (no random perturbations)
    if len(pit_env_ids) > 0:
        root_states = asset.data.default_root_state[pit_env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[pit_env_ids]
        orientations = root_states[:, 3:7]
        velocities = torch.zeros_like(root_states[:, 7:13])
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=pit_env_ids)

    # Reset non-pit environments with random perturbations
    if len(non_pit_env_ids) > 0:
        root_states = asset.data.default_root_state[non_pit_env_ids].clone()

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        positions = root_states[:, 0:3] + env.scene.env_origins[non_pit_env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=non_pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=non_pit_env_ids)


def reset_root_state_fallen(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (-0.5, 0.5),
    joint_velocity_range: tuple[float, float] = (-1.0, 1.0),
    fallen_probability: float = 1.0,
    ground_height_offset: float = 0.05,
    allow_random_orientation: bool = True,
    simple_fall_type: str | None = None,
    body_half_extents: tuple[float, float, float] = (0.22, 0.16, 0.09),
    spawn_height_margin: float = 0.05,
    max_fallen_spawn_height: float | None = None,
    clamp_joint_positions: bool = True,
    reject_near_upright_fallen: bool = True,
    near_upright_projected_gravity_z: float = -0.55,
    use_recovery_curriculum: bool = False,
):
    """Reset the robot to a random fallen pose for stand-up recovery training.

    This function samples random orientations (roll/pitch in fallen ranges) and
    random joint positions, places the robot near the ground, and sets random
    velocities to create diverse initial conditions for recovery learning.

    Fall type distribution:
      - Side fall: large roll (±60° to ±90°), moderate pitch
      - Front/back fall: large pitch (±60° to ±90°), moderate roll
      - Tilted: moderate roll and pitch (30° to 60°)
      - Normal (10%): near-upright for curriculum mixing

    Args:
        env: The simulation environment.
        env_ids: Environment indices to reset.
        asset_cfg: The asset configuration.
        pose_range: Ranges for root pose randomization. If None, uses fallen defaults.
        velocity_range: Ranges for root velocity randomization. If None, uses defaults.
        joint_position_range: (min, max) range for random joint positions [rad].
        joint_velocity_range: (min, max) range for random joint velocities [rad/s].
        fallen_probability: Probability that a given env gets a fallen pose (vs normal).
        ground_height_offset: Extra height offset above ground for the base [m].
        body_half_extents: Approximate base-body half extents used to keep random orientations above ground.
        spawn_height_margin: Extra clearance above the oriented body support radius.
        max_fallen_spawn_height: Optional cap on fallen spawn height.
        clamp_joint_positions: Clamp randomized joints to soft joint position limits when available.
        reject_near_upright_fallen: Replace fallen samples that are already too close to upright.
        near_upright_projected_gravity_z: Fallen samples below this projected gravity z are considered too easy.
        use_recovery_curriculum: Adjust fallen pose difficulty from env._recovery_curriculum_stage.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Resolve joint indices: use all joints if not specified
    raw_ids = asset_cfg.joint_ids
    if raw_ids is None or (isinstance(raw_ids, slice) and raw_ids == slice(None)):
        joint_ids = list(range(asset.num_joints))
    elif isinstance(raw_ids, slice):
        joint_ids = list(range(asset.num_joints))[raw_ids]
    else:
        joint_ids = list(raw_ids)
    num_joints = len(joint_ids)

    if pose_range is None:
        pose_range = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
            "roll": (-3.14, 3.14),
            "pitch": (-3.14, 3.14),
            "yaw": (-3.14, 3.14),
        }
    if velocity_range is None:
        velocity_range = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }

    device = asset.device
    root_states = asset.data.default_root_state[env_ids].clone()
    recovery_stage = int(getattr(env, "_recovery_curriculum_stage", 0)) if use_recovery_curriculum else 3

    # --- Determine which envs get fallen vs normal ---
    is_fallen = torch.rand(num_envs, device=device) < fallen_probability

    # --- Sample root orientations ---
    # Fallen orientations: large roll or pitch
    roll = torch.zeros(num_envs, device=device)
    pitch = torch.zeros(num_envs, device=device)
    yaw = torch.zeros(num_envs, device=device)

    if is_fallen.any():
        n_fallen = is_fallen.sum().item()
        if n_fallen > 0:
            fallen_ids = torch.where(is_fallen)[0]

            if simple_fall_type == "pitch_positive":
                roll[fallen_ids] = math_utils.sample_uniform(0.10, 0.28, (n_fallen,), device=device)
                pitch[fallen_ids] = math_utils.sample_uniform(1.15, 1.45, (n_fallen,), device=device)
                yaw[fallen_ids] = math_utils.sample_uniform(-0.05, 0.05, (n_fallen,), device=device)
                fall_type = None
            elif simple_fall_type == "pitch_negative":
                roll[fallen_ids] = math_utils.sample_uniform(0.10, 0.28, (n_fallen,), device=device)
                pitch[fallen_ids] = math_utils.sample_uniform(-1.45, -1.15, (n_fallen,), device=device)
                yaw[fallen_ids] = math_utils.sample_uniform(-0.05, 0.05, (n_fallen,), device=device)
                fall_type = None
            elif use_recovery_curriculum and recovery_stage <= 0:
                fall_type = torch.randint(0, 2, (n_fallen,), device=device)
            elif use_recovery_curriculum and recovery_stage == 1:
                fall_type = torch.multinomial(torch.tensor([0.45, 0.45, 0.10], device=device), n_fallen, replacement=True)
            elif use_recovery_curriculum and recovery_stage == 2:
                fall_type = torch.multinomial(torch.tensor([0.35, 0.35, 0.30], device=device), n_fallen, replacement=True)
            else:
                fall_type = torch.randint(0, 3, (n_fallen,), device=device)

            # Three fall types with equal probability. The self-righting
            # curriculum can disable fully random orientations for its first
            # stage, replacing them with a moderate diagonal fall.

            # Type 0: Side fall (large roll, small pitch)
            mask = (fall_type == 0) if fall_type is not None else torch.zeros(n_fallen, dtype=torch.bool, device=device)
            if mask.any():
                # Roll: ±60° to ±100° (1.05 to 1.75 rad), random sign
                if use_recovery_curriculum and recovery_stage <= 0:
                    roll_range = (1.10, 1.45)
                    pitch_range = (-0.25, 0.25)
                elif use_recovery_curriculum and recovery_stage == 1:
                    roll_range = (1.15, 1.75)
                    pitch_range = (-0.45, 0.45)
                else:
                    roll_range = (1.05, 1.75)
                    pitch_range = (-0.5, 0.5)
                roll_mag = math_utils.sample_uniform(*roll_range, (mask.sum().item(),), device=device)
                roll_sign = torch.sign(math_utils.sample_uniform(-1.0, 1.0, (mask.sum().item(),), device=device))
                roll[fallen_ids[mask]] = roll_mag * roll_sign
                pitch[fallen_ids[mask]] = math_utils.sample_uniform(*pitch_range, (mask.sum().item(),), device=device)
                yaw[fallen_ids[mask]] = math_utils.sample_uniform(-3.14, 3.14, (mask.sum().item(),), device=device)

            # Type 1: Front/back fall (large pitch, small to moderate roll)
            mask = (fall_type == 1) if fall_type is not None else torch.zeros(n_fallen, dtype=torch.bool, device=device)
            if mask.any():
                if use_recovery_curriculum and recovery_stage <= 0:
                    pitch_range = (1.10, 1.45)
                    roll_range = (-0.25, 0.25)
                elif use_recovery_curriculum and recovery_stage == 1:
                    pitch_range = (1.15, 1.75)
                    roll_range = (-0.55, 0.55)
                else:
                    pitch_range = (1.05, 1.75)
                    roll_range = (-0.8, 0.8)
                pitch_mag = math_utils.sample_uniform(*pitch_range, (mask.sum().item(),), device=device)
                pitch_sign = torch.sign(math_utils.sample_uniform(-1.0, 1.0, (mask.sum().item(),), device=device))
                pitch[fallen_ids[mask]] = pitch_mag * pitch_sign
                roll[fallen_ids[mask]] = math_utils.sample_uniform(*roll_range, (mask.sum().item(),), device=device)
                yaw[fallen_ids[mask]] = math_utils.sample_uniform(-3.14, 3.14, (mask.sum().item(),), device=device)

            # Type 2: Completely random (any orientation, can be upside-down)
            mask = (fall_type == 2) if fall_type is not None else torch.zeros(n_fallen, dtype=torch.bool, device=device)
            if mask.any():
                if allow_random_orientation and (not use_recovery_curriculum or recovery_stage >= 2):
                    roll[fallen_ids[mask]] = math_utils.sample_uniform(-3.14, 3.14, (mask.sum().item(),), device=device)
                    pitch[fallen_ids[mask]] = math_utils.sample_uniform(-3.14, 3.14, (mask.sum().item(),), device=device)
                else:
                    roll_mag = math_utils.sample_uniform(0.7, 1.2, (mask.sum().item(),), device=device)
                    pitch_mag = math_utils.sample_uniform(0.7, 1.2, (mask.sum().item(),), device=device)
                    roll_sign = torch.sign(math_utils.sample_uniform(-1.0, 1.0, (mask.sum().item(),), device=device))
                    pitch_sign = torch.sign(math_utils.sample_uniform(-1.0, 1.0, (mask.sum().item(),), device=device))
                    roll[fallen_ids[mask]] = roll_mag * roll_sign
                    pitch[fallen_ids[mask]] = pitch_mag * pitch_sign
                yaw[fallen_ids[mask]] = math_utils.sample_uniform(-3.14, 3.14, (mask.sum().item(),), device=device)

    # Normal envs: small perturbations around upright
    normal_ids = torch.where(~is_fallen)[0]
    if len(normal_ids) > 0:
        roll[normal_ids] = math_utils.sample_uniform(-0.1, 0.1, (len(normal_ids),), device=device)
        pitch[normal_ids] = math_utils.sample_uniform(-0.1, 0.1, (len(normal_ids),), device=device)
        yaw[normal_ids] = math_utils.sample_uniform(-3.14, 3.14, (len(normal_ids),), device=device)

    orientations_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    if reject_near_upright_fallen and is_fallen.any():
        quat = orientations / torch.clamp(torch.linalg.norm(orientations, dim=1, keepdim=True), min=1.0e-6)
        qx, qy = quat[:, 1], quat[:, 2]
        projected_gravity_z = -(1.0 - 2.0 * (qx * qx + qy * qy))
        easy_fallen = is_fallen & (projected_gravity_z < near_upright_projected_gravity_z)
        if easy_fallen.any():
            easy_count = int(easy_fallen.sum().item())
            easy_ids = torch.where(easy_fallen)[0]
            replacement_type = torch.randint(0, 2, (easy_count,), device=device)

            side_mask = replacement_type == 0
            if side_mask.any():
                count = int(side_mask.sum().item())
                roll_mag = math_utils.sample_uniform(1.25, 1.95, (count,), device=device)
                roll_sign = torch.sign(math_utils.sample_uniform(-1.0, 1.0, (count,), device=device))
                ids = easy_ids[side_mask]
                roll[ids] = roll_mag * roll_sign
                pitch[ids] = math_utils.sample_uniform(-0.65, 0.65, (count,), device=device)

            pitch_mask = ~side_mask
            if pitch_mask.any():
                count = int(pitch_mask.sum().item())
                pitch_mag = math_utils.sample_uniform(1.25, 1.95, (count,), device=device)
                pitch_sign = torch.sign(math_utils.sample_uniform(-1.0, 1.0, (count,), device=device))
                ids = easy_ids[pitch_mask]
                pitch[ids] = pitch_mag * pitch_sign
                roll[ids] = math_utils.sample_uniform(-0.75, 0.75, (count,), device=device)

            yaw[easy_ids] = math_utils.sample_uniform(-3.14, 3.14, (easy_count,), device=device)
            orientations_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
            orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # --- Sample root positions ---
    pos_x = math_utils.sample_uniform(
        pose_range["x"][0], pose_range["x"][1], (num_envs,), device=device
    )
    pos_y = math_utils.sample_uniform(
        pose_range["y"][0], pose_range["y"][1], (num_envs,), device=device
    )
    # Base height: close to ground for fallen, but high enough that the randomly
    # oriented base body is not spawned deeply inside the plane.
    quat = orientations
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=1, keepdim=True), min=1.0e-6)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot_z_row_abs = torch.stack(
        [
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dim=1,
    ).abs()
    half_extents = torch.tensor(body_half_extents, device=device, dtype=root_states.dtype)
    oriented_body_z_radius = torch.sum(rot_z_row_abs * half_extents.unsqueeze(0), dim=1)
    pos_z_fallen = torch.clamp(
        oriented_body_z_radius + spawn_height_margin,
        min=ground_height_offset,
    )
    if max_fallen_spawn_height is not None:
        pos_z_fallen = torch.clamp(pos_z_fallen, max=max_fallen_spawn_height)
    pos_z_normal = root_states[:, 2]  # default height
    pos_z = torch.where(is_fallen, pos_z_fallen, pos_z_normal)

    positions = root_states[:, 0:3].clone()
    positions[:, 0] += pos_x
    positions[:, 1] += pos_y
    positions[:, 2] = pos_z
    positions += env.scene.env_origins[env_ids]

    # --- Sample root velocities ---
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=device)
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (num_envs, 6), device=device
    )
    velocities = root_states[:, 7:13] + vel_samples

    # --- Sample random joint positions ---
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    jp_min, jp_max = joint_position_range
    joint_noise = math_utils.sample_uniform(jp_min, jp_max, (num_envs, num_joints), device=device)
    joint_pos[:, joint_ids] += joint_noise
    if clamp_joint_positions and hasattr(asset.data, "soft_joint_pos_limits"):
        joint_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
        lower = joint_limits[..., 0]
        upper = joint_limits[..., 1]
        valid_limits = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
        values = joint_pos[:, joint_ids]
        clamped_values = torch.minimum(torch.maximum(values, lower), upper)
        joint_pos[:, joint_ids] = torch.where(valid_limits, clamped_values, values)

    # --- Sample random joint velocities ---
    jv_min, jv_max = joint_velocity_range
    joint_vel = math_utils.sample_uniform(jv_min, jv_max, (num_envs, num_joints), device=device)

    # --- Write to simulation ---
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    asset.write_joint_state_to_sim(
        joint_pos[:, joint_ids],
        joint_vel,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
