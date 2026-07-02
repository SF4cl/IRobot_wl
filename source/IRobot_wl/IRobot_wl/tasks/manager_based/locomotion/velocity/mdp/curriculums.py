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


def _stage_threshold(values: Sequence[float], stage: int, default: float) -> float:
    if len(values) == 0:
        return default
    index = min(max(int(stage), 0), len(values) - 1)
    return float(values[index])


def _advance_stage_window(
    env: ManagerBasedRLEnv,
    prefix: str,
    passed: bool,
    pass_windows: int,
) -> bool:
    count_name = f"_{prefix}_pass_windows"
    current = int(getattr(env, count_name, 0))
    current = current + 1 if passed else 0
    setattr(env, count_name, current)
    return current >= max(int(pass_windows), 1)


def _reset_metric_window(env: ManagerBasedRLEnv, prefix: str):
    for name in ("count", "upright_sum", "ready_sum", "still_sum", "height_error_sum", "lin_error_sum", "ang_error_sum"):
        setattr(env, f"_{prefix}_{name}", 0.0)


def _accumulate_recovery_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    prefix: str,
    theta0_ready_threshold: float = 0.45,
    lin_vel_threshold: float = 0.35,
    ang_vel_threshold: float = 0.7,
):
    if env.common_step_counter <= 0 or len(env_ids) == 0:
        return

    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    env_ids_tensor = torch.as_tensor(env_ids, device=device, dtype=torch.long)

    projected_gravity = robot.data.projected_gravity_b[env_ids_tensor]
    upright = projected_gravity[:, 2] < -0.75
    base_lin_vel = robot.data.root_lin_vel_b[env_ids_tensor]
    base_ang_vel = robot.data.root_ang_vel_b[env_ids_tensor]
    still = (
        torch.linalg.norm(base_lin_vel[:, :2], dim=1) < lin_vel_threshold
    ) & (torch.abs(base_ang_vel[:, 2]) < ang_vel_threshold)

    try:
        from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state

        vmc_cfg = env.cfg.vmc_actions
        leg_joint_indices = robot.find_joints(
            ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"], preserve_order=True
        )[0]
        wheel_joint_indices = robot.find_joints(["l_wheel_Joint", "r_wheel_Joint"], preserve_order=True)[0]
        state = compute_vmc_state(
            robot.data.joint_pos,
            robot.data.joint_vel,
            list(leg_joint_indices),
            list(wheel_joint_indices),
            vmc_cfg.l1,
            vmc_cfg.l2,
            vmc_cfg.offset,
            vmc_cfg.theta1_offset,
            vmc_cfg.theta2_offset,
            env.step_dt,
        )
        theta_ready = state["theta0"][env_ids_tensor].abs().lt(theta0_ready_threshold).all(dim=1)
    except Exception:
        theta_ready = torch.ones_like(upright)

    command = env.command_manager.get_command("base_velocity")[env_ids_tensor]
    command_term = env.command_manager.get_term("base_velocity")
    base_height_cmd = getattr(command_term, "base_height_command_b", None)
    if base_height_cmd is None:
        base_height_cmd = torch.full((env.num_envs,), 0.235, device=device)
    height_error = (robot.data.root_pos_w[env_ids_tensor, 2] - base_height_cmd[env_ids_tensor]).abs()
    lin_error = (command[:, 0] - base_lin_vel[:, 0]).abs()
    ang_error = (command[:, 2] - base_ang_vel[:, 2]).abs()

    ready = upright & theta_ready
    count = float(len(env_ids_tensor))
    setattr(env, f"_{prefix}_count", float(getattr(env, f"_{prefix}_count", 0.0)) + count)
    setattr(env, f"_{prefix}_upright_sum", float(getattr(env, f"_{prefix}_upright_sum", 0.0)) + float(upright.float().sum().item()))
    setattr(env, f"_{prefix}_ready_sum", float(getattr(env, f"_{prefix}_ready_sum", 0.0)) + float(ready.float().sum().item()))
    setattr(env, f"_{prefix}_still_sum", float(getattr(env, f"_{prefix}_still_sum", 0.0)) + float(still.float().sum().item()))
    setattr(env, f"_{prefix}_height_error_sum", float(getattr(env, f"_{prefix}_height_error_sum", 0.0)) + float(height_error.sum().item()))
    setattr(env, f"_{prefix}_lin_error_sum", float(getattr(env, f"_{prefix}_lin_error_sum", 0.0)) + float(lin_error.sum().item()))
    setattr(env, f"_{prefix}_ang_error_sum", float(getattr(env, f"_{prefix}_ang_error_sum", 0.0)) + float(ang_error.sum().item()))


def _metric_window_values(env: ManagerBasedRLEnv, prefix: str) -> dict[str, float] | None:
    count = float(getattr(env, f"_{prefix}_count", 0.0))
    if count <= 0.0:
        return None
    return {
        "upright": float(getattr(env, f"_{prefix}_upright_sum", 0.0)) / count,
        "ready": float(getattr(env, f"_{prefix}_ready_sum", 0.0)) / count,
        "still": float(getattr(env, f"_{prefix}_still_sum", 0.0)) / count,
        "height_error": float(getattr(env, f"_{prefix}_height_error_sum", 0.0)) / count,
        "lin_error": float(getattr(env, f"_{prefix}_lin_error_sum", 0.0)) / count,
        "ang_error": float(getattr(env, f"_{prefix}_ang_error_sum", 0.0)) / count,
    }


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
    performance_gate: bool = False,
    min_stage_iterations: Sequence[int] = (1200, 1800, 2200, 2200),
    eval_interval_iterations: int = 120,
    pass_windows: int = 3,
    min_completed_episodes: int = 256,
    upright_success_thresholds: Sequence[float] = (0.62, 0.68, 0.72, 0.78),
    ready_success_thresholds: Sequence[float] = (0.10, 0.24, 0.42, 0.62),
    still_success_thresholds: Sequence[float] = (0.0, 0.0, 0.25, 0.58),
    max_height_error_thresholds: Sequence[float] = (1.0, 1.0, 0.060, 0.040),
) -> torch.Tensor:
    """Expose a simple global recovery stage for resets and reward gates.

    Stages:
      0: easy self-righting from side/front/back falls.
      1: harder self-righting and wheel-under-body stand-up.
      2: mixed random orientations, still dominated by recovery.
      3: stage2.5 static stabilization after recovery.
      4: full recovery-to-locomotion objective.
    """
    iteration_offset = int(getattr(env, iteration_offset_attr, 0))
    learning_iteration = int(env.common_step_counter // max(int(rollout_steps_per_iteration), 1)) + iteration_offset
    if len(stage_steps) < 4:
        raise ValueError("stage_steps must contain four iteration thresholds.")

    if performance_gate:
        prefix = "recovery_stage_perf"
        if getattr(env, "_recovery_perf_iteration_offset", None) != iteration_offset:
            initial_stage = int(getattr(env, "_recovery_curriculum_stage", 0))
            setattr(env, "_recovery_perf_iteration_offset", iteration_offset)
            setattr(env, "_recovery_curriculum_stage", initial_stage)
            setattr(env, "_recovery_stage_start_iteration", learning_iteration)
            setattr(env, "_recovery_stage_last_eval_iteration", learning_iteration)
            setattr(env, "_recovery_stage_perf_pass_windows", 0)
            _reset_metric_window(env, prefix)

        stage = int(getattr(env, "_recovery_curriculum_stage", 0))
        stage = min(max(stage, 0), 4)
        _accumulate_recovery_metrics(env, env_ids, prefix)

        if stage < 4:
            last_eval = int(getattr(env, "_recovery_stage_last_eval_iteration", learning_iteration))
            stage_start = int(getattr(env, "_recovery_stage_start_iteration", learning_iteration))
            interval = max(int(eval_interval_iterations), 1)
            stage_elapsed = learning_iteration - stage_start
            if learning_iteration - last_eval >= interval:
                metrics = _metric_window_values(env, prefix)
                count = float(getattr(env, f"_{prefix}_count", 0.0))
                min_elapsed = _stage_threshold(min_stage_iterations, stage, 0.0)
                passed = False
                if metrics is not None and count >= float(min_completed_episodes) and stage_elapsed >= min_elapsed:
                    passed = (
                        metrics["upright"] >= _stage_threshold(upright_success_thresholds, stage, 0.75)
                        and metrics["ready"] >= _stage_threshold(ready_success_thresholds, stage, 0.50)
                        and metrics["still"] >= _stage_threshold(still_success_thresholds, stage, 0.0)
                        and metrics["height_error"] <= _stage_threshold(max_height_error_thresholds, stage, 0.05)
                    )
                if _advance_stage_window(env, "recovery_stage_perf", passed, pass_windows):
                    stage += 1
                    setattr(env, "_recovery_curriculum_stage", stage)
                    setattr(env, "_recovery_stage_start_iteration", learning_iteration)
                    setattr(env, "_recovery_stage_perf_pass_windows", 0)
                    if stage == 4 and not hasattr(env, "_recovery_locomotion_substage"):
                        setattr(env, "_recovery_locomotion_substage", 0)
                setattr(env, "_recovery_stage_last_eval_iteration", learning_iteration)
                _reset_metric_window(env, prefix)

        env._recovery_curriculum_stage = stage
        return torch.tensor(float(stage), device=env.device)

    del env_ids
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


def recovery_locomotion_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    lin_vel_ranges: Sequence[Sequence[float]] = ((-0.45, 0.45), (-0.9, 0.9), (-1.5, 1.5), (-1.5, 1.5)),
    ang_vel_ranges: Sequence[Sequence[float]] = ((0.0, 0.0), (-0.35, 0.35), (-0.75, 0.75), (-1.0, 1.0)),
    height_ranges: Sequence[Sequence[float]] = ((0.21, 0.24), (0.20, 0.26), (0.19, 0.28), (0.19, 0.28)),
    min_substage_iterations: Sequence[int] = (1000, 1200, 1400),
    eval_interval_iterations: int = 120,
    rollout_steps_per_iteration: int = 96,
    pass_windows: int = 3,
    min_completed_episodes: int = 256,
    lin_error_thresholds: Sequence[float] = (0.18, 0.22, 0.26),
    ang_error_thresholds: Sequence[float] = (0.18, 0.24, 0.30),
    height_error_thresholds: Sequence[float] = (0.025, 0.025, 0.020),
    upright_success_threshold: float = 0.80,
    ready_success_threshold: float = 0.65,
    iteration_offset_attr: str = "_recovery_curriculum_iteration_offset",
    reset_existing_commands: bool = True,
) -> torch.Tensor:
    """Open locomotion command difficulty only after stage4 performance is stable."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges_cfg = command_term.cfg.ranges
    height_cfg = command_term.cfg
    stage = int(getattr(env, "_recovery_curriculum_stage", 0))
    prefix = "recovery_locomotion_perf"
    iteration_offset = int(getattr(env, iteration_offset_attr, 0))
    learning_iteration = int(env.common_step_counter // max(int(rollout_steps_per_iteration), 1)) + iteration_offset

    if getattr(env, "_recovery_locomotion_iteration_offset", None) != iteration_offset:
        setattr(env, "_recovery_locomotion_iteration_offset", iteration_offset)
        setattr(env, "_recovery_locomotion_substage", int(getattr(env, "_recovery_locomotion_substage", 0)))
        setattr(env, "_recovery_locomotion_start_iteration", learning_iteration)
        setattr(env, "_recovery_locomotion_last_eval_iteration", learning_iteration)
        setattr(env, "_recovery_locomotion_perf_pass_windows", 0)
        _reset_metric_window(env, prefix)

    if stage < 4:
        substage = 0
        lin_range = (0.0, 0.0)
        ang_range = (0.0, 0.0)
        height_range = tuple(float(v) for v in height_ranges[0])
    else:
        substage = min(max(int(getattr(env, "_recovery_locomotion_substage", 0)), 0), len(lin_vel_ranges) - 1)
        _accumulate_recovery_metrics(env, env_ids, prefix, theta0_ready_threshold=0.45)
        if substage < len(lin_vel_ranges) - 1:
            last_eval = int(getattr(env, "_recovery_locomotion_last_eval_iteration", learning_iteration))
            substage_start = int(getattr(env, "_recovery_locomotion_start_iteration", learning_iteration))
            interval = max(int(eval_interval_iterations), 1)
            substage_elapsed = learning_iteration - substage_start
            if learning_iteration - last_eval >= interval:
                metrics = _metric_window_values(env, prefix)
                count = float(getattr(env, f"_{prefix}_count", 0.0))
                min_elapsed = _stage_threshold(min_substage_iterations, substage, 0.0)
                passed = False
                if metrics is not None and count >= float(min_completed_episodes) and substage_elapsed >= min_elapsed:
                    passed = (
                        metrics["upright"] >= float(upright_success_threshold)
                        and metrics["ready"] >= float(ready_success_threshold)
                        and metrics["lin_error"] <= _stage_threshold(lin_error_thresholds, substage, 0.25)
                        and metrics["ang_error"] <= _stage_threshold(ang_error_thresholds, substage, 0.25)
                        and metrics["height_error"] <= _stage_threshold(height_error_thresholds, substage, 0.025)
                    )
                if _advance_stage_window(env, "recovery_locomotion_perf", passed, pass_windows):
                    substage += 1
                    setattr(env, "_recovery_locomotion_substage", substage)
                    setattr(env, "_recovery_locomotion_start_iteration", learning_iteration)
                    setattr(env, "_recovery_locomotion_perf_pass_windows", 0)
                setattr(env, "_recovery_locomotion_last_eval_iteration", learning_iteration)
                _reset_metric_window(env, prefix)

        lin_range = tuple(float(v) for v in lin_vel_ranges[substage])
        ang_range = tuple(float(v) for v in ang_vel_ranges[substage])
        height_range = tuple(float(v) for v in height_ranges[substage])

    old_height_range = tuple(float(v) for v in getattr(height_cfg, "base_height_range", height_range))
    ranges_cfg.lin_vel_x = lin_range
    ranges_cfg.lin_vel_y = (0.0, 0.0)
    ranges_cfg.ang_vel_z = ang_range
    height_cfg.base_height_range = height_range

    if reset_existing_commands and hasattr(command_term, "base_height_command_b"):
        command_term.base_height_command_b.clamp_(min=height_range[0], max=height_range[1])

    if hasattr(command_term, "vel_command_b"):
        if stage < 4:
            command_term.vel_command_b[:, :] = 0.0
            if hasattr(command_term, "is_standing_env"):
                command_term.is_standing_env[:] = True
            if hasattr(command_term, "is_heading_env"):
                command_term.is_heading_env[:] = False
        else:
            command_term.vel_command_b[:, 0].clamp_(min=lin_range[0], max=lin_range[1])
            command_term.vel_command_b[:, 1] = 0.0
            command_term.vel_command_b[:, 2].clamp_(min=ang_range[0], max=ang_range[1])

    setattr(env, "_recovery_locomotion_substage", int(substage))
    return torch.tensor(float(substage), device=env.device)
