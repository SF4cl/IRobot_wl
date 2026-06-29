"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _completed_episode_score(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    update_interval: int,
    state_prefix: str,
) -> torch.Tensor | None:
    """Average completed-episode reward rate when the update interval elapses."""
    sum_name = f"_{state_prefix}_score_sum"
    count_name = f"_{state_prefix}_score_count"
    step_name = f"_{state_prefix}_last_update_step"

    if not hasattr(env, sum_name):
        setattr(env, sum_name, torch.zeros((), device=env.device))
        setattr(env, count_name, 0)
        setattr(env, step_name, env.common_step_counter)

    if env.common_step_counter > 0:
        episode_steps = env.episode_length_buf[env_ids].clamp_min(1)
        observed_steps = torch.minimum(
            episode_steps,
            torch.full_like(episode_steps, env.common_step_counter),
        )
        episode_time = observed_steps * env.step_dt
        episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
        setattr(env, sum_name, getattr(env, sum_name) + torch.sum(episode_sums / episode_time))
        setattr(env, count_name, getattr(env, count_name) + episode_sums.numel())

    elapsed_steps = env.common_step_counter - getattr(env, step_name)
    if elapsed_steps < update_interval or getattr(env, count_name) == 0:
        return None

    score = getattr(env, sum_name) / getattr(env, count_name)
    getattr(env, sum_name).zero_()
    setattr(env, count_name, 0)
    setattr(env, step_name, env.common_step_counter)
    return score


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    threshold: float = 0.8,
    step_size: float = 0.1,
    update_interval_s: float | None = None,
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges only once. Curriculum terms can be evaluated
    # more than once while common_step_counter is still 0 during environment
    # setup, so common_step_counter alone would repeatedly shrink the range.
    if not hasattr(env, "_original_vel_x"):
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    update_interval = env.max_episode_length
    if update_interval_s is not None:
        update_interval = max(1, int(round(update_interval_s / env.step_dt)))

    score = _completed_episode_score(
        env,
        env_ids,
        reward_term_name,
        update_interval,
        state_prefix="lin_vel_curriculum",
    )
    if score is not None:
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

        # If the tracking reward is above the configured threshold, increase the command range.
        if score > threshold * reward_term_cfg.weight:
            new_vel_x = vel_x + torch.sign(env._final_vel_x - vel_x) * step_size
            new_vel_y = vel_y + torch.sign(env._final_vel_y - vel_y) * step_size

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.minimum(torch.maximum(new_vel_x, torch.minimum(vel_x, env._final_vel_x)), torch.maximum(vel_x, env._final_vel_x))
            new_vel_y = torch.minimum(torch.maximum(new_vel_y, torch.minimum(vel_y, env._final_vel_y)), torch.maximum(vel_y, env._final_vel_y))

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    threshold: float = 0.8,
    step_size: float = 0.1,
    update_interval_s: float | None = None,
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity range only once. See the linear velocity
    # curriculum above for why common_step_counter == 0 is not sufficient.
    if not hasattr(env, "_original_ang_vel_z"):
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    update_interval = env.max_episode_length
    if update_interval_s is not None:
        update_interval = max(1, int(round(update_interval_s / env.step_dt)))

    score = _completed_episode_score(
        env,
        env_ids,
        reward_term_name,
        update_interval,
        state_prefix="ang_vel_curriculum",
    )
    if score is not None:
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)

        # If the tracking reward is above the configured threshold, increase the command range.
        if score > threshold * reward_term_cfg.weight:
            new_ang_vel_z = ang_vel_z + torch.sign(env._final_ang_vel_z - ang_vel_z) * step_size

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.minimum(
                torch.maximum(new_ang_vel_z, torch.minimum(ang_vel_z, env._final_ang_vel_z)),
                torch.maximum(ang_vel_z, env._final_ang_vel_z),
            )

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def recovery_staged_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    stage_steps: Sequence[int] = (300, 900, 1500, 2100),
    rollout_steps_per_iteration: int = 96,
    iteration_offset_attr: str = "_recovery_curriculum_iteration_offset",
) -> torch.Tensor:
    """Expose a simple global recovery stage for resets and reward gates.

    Stages:
      0: easy self-righting from side/front/back falls.
      1: harder self-righting and wheel-under-body stand-up.
      2: mixed random orientations, still dominated by recovery.
      3: stage2.5 static stabilization after recovery.
      4: full recovery-to-locomotion objective.
    """
    del env_ids
    iteration_offset = int(getattr(env, iteration_offset_attr, 0))
    learning_iteration = int(env.common_step_counter // max(int(rollout_steps_per_iteration), 1)) + iteration_offset
    if len(stage_steps) < 4:
        raise ValueError("stage_steps must contain four iteration thresholds.")

    if learning_iteration < stage_steps[0]:
        stage = 0
    elif learning_iteration < stage_steps[1]:
        stage = 1
    elif learning_iteration < stage_steps[2]:
        stage = 2
    elif learning_iteration < stage_steps[3]:
        stage = 3
    else:
        stage = 4

    env._recovery_curriculum_stage = stage
    return torch.tensor(float(stage), device=env.device)


def command_levels_base_height(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    initial_range: Sequence[float] = (0.19, 0.22),
    final_range: Sequence[float] = (0.19, 0.28),
    step_size: float = 0.02,
    update_interval_iterations: int = 300,
    rollout_steps_per_iteration: int = 96,
    iteration_offset_attr: str = "_recovery_curriculum_iteration_offset",
    reset_existing_commands: bool = True,
) -> torch.Tensor:
    """Gradually expand the sampled base-height command range during stage4 training."""
    del env_ids
    command_term = env.command_manager.get_term("base_velocity")
    cfg = command_term.cfg

    if len(initial_range) != 2 or len(final_range) != 2:
        raise ValueError("initial_range and final_range must contain two values.")

    initial = torch.tensor(initial_range, device=env.device, dtype=torch.float32)
    final = torch.tensor(final_range, device=env.device, dtype=torch.float32)
    if torch.any(initial > final):
        raise ValueError("initial_range must not exceed final_range elementwise.")

    iteration_offset = int(getattr(env, iteration_offset_attr, 0))
    learning_iteration = int(env.common_step_counter // max(int(rollout_steps_per_iteration), 1)) + iteration_offset
    offset_name = "_base_height_curriculum_iteration_offset"
    start_name = "_base_height_curriculum_stage4_start"
    previous_offset = getattr(env, offset_name, None)
    if previous_offset != iteration_offset:
        setattr(env, offset_name, iteration_offset)
        setattr(env, start_name, iteration_offset)
    stage4_start = int(getattr(env, start_name, iteration_offset))

    interval = max(int(update_interval_iterations), 1)
    levels = max((learning_iteration - stage4_start) // interval, 0)
    new_range = torch.minimum(initial + levels * float(step_size), final)
    range_tuple = (float(new_range[0].item()), float(new_range[1].item()))
    old_range = tuple(float(x) for x in getattr(cfg, "base_height_range", final_range))
    cfg.base_height_range = range_tuple

    if reset_existing_commands and old_range != range_tuple and hasattr(command_term, "base_height_command_b"):
        command_term.base_height_command_b.clamp_(min=range_tuple[0], max=range_tuple[1])

    return torch.tensor(range_tuple[1], device=env.device)
