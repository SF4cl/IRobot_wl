from __future__ import annotations

import json
import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from .actor_critic_sequence import ActorCriticSequence
from .ppo import PPO


class WlSequenceRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        print("[DEBUG][WlSequenceRunner] init start")
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir
        self.writer = None
        self._diagnostics_path = os.path.join(log_dir, "diagnostics.jsonl") if log_dir is not None else None
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.git_status_repos = []
        self._robot = self.env.unwrapped.scene["robot"]
        self._leg_joint_ids, _ = self._robot.find_joints(["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"], preserve_order=True)
        self._wheel_joint_ids, _ = self._robot.find_joints(["l_wheel_Joint", "r_wheel_Joint"], preserve_order=True)
        self._prev_debug_commands = None
        self._prev_debug_base_lin_vel = None
        self._prev_debug_base_ang_vel = None
        self._steps_since_command_change = None

        policy_cfg = train_cfg["policy"]
        actor_group = train_cfg.get("actor_obs_group", "policy")
        critic_group = train_cfg.get("critic_obs_group", "critic")
        history_group = train_cfg.get("obs_history_group", "policy_history")
        print("[DEBUG][WlSequenceRunner] fetching observations...")
        obs = self.env.get_observations()
        print("[DEBUG][WlSequenceRunner] observations fetched")
        num_obs = obs.get(actor_group).shape[1]
        num_critic_obs = obs.get(critic_group).shape[1] + policy_cfg["latent_dim"]
        num_encoder_obs = obs.get(history_group).shape[1]
        num_actions = self.env.num_actions
        self.actor_group = actor_group
        self.critic_group = critic_group
        self.history_group = history_group

        actor_critic = ActorCriticSequence(
            num_obs=num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_encoder_obs=num_encoder_obs,
            latent_dim=policy_cfg["latent_dim"],
            encoder_hidden_dims=policy_cfg["encoder_hidden_dims"],
            actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            critic_hidden_dims=policy_cfg["critic_hidden_dims"],
            activation=policy_cfg["activation"],
            init_noise_std=policy_cfg["init_noise_std"],
        ).to(self.device)
        print(
            f"[DEBUG][WlSequenceRunner] dims actor={num_obs} critic={num_critic_obs} "
            f"history={num_encoder_obs} actions={num_actions}"
        )
        algo_cfg = self._sanitize_algorithm_cfg(train_cfg["algorithm"])
        self.alg = PPO(actor_critic, device=device, **algo_cfg)
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [num_encoder_obs],
            [num_actions],
        )
        print("[DEBUG][WlSequenceRunner] storage initialized, resetting env...")
        self.env.reset()
        print("[DEBUG][WlSequenceRunner] init done")

    @staticmethod
    def _sanitize_algorithm_cfg(algo_cfg: dict) -> dict:
        allowed_keys = {
            "num_learning_epochs",
            "num_mini_batches",
            "clip_param",
            "gamma",
            "lam",
            "value_loss_coef",
            "entropy_coef",
            "learning_rate",
            "extra_learning_rate",
            "max_grad_norm",
            "use_clipped_value_loss",
            "schedule",
            "desired_kl",
        }
        return {key: value for key, value in algo_cfg.items() if key in allowed_keys}

    def add_git_repo_to_log(self, repo_file_path: str):
        self.git_status_repos.append(repo_file_path)

    def _split_obs(self, obs):
        return (
            obs.get(self.actor_group).to(self.device),
            obs.get(self.history_group).to(self.device),
            obs.get(self.critic_group).to(self.device),
        )

    def _format_tensor(self, tensor: torch.Tensor, precision: int = 3) -> str:
        values = [f"{v:.{precision}f}" for v in tensor.detach().cpu().tolist()]
        return "[" + ", ".join(values) + "]"

    def _as_float(self, value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu().item())
        return float(value)

    def _as_list(self, tensor: torch.Tensor) -> list[float]:
        return [float(v) for v in tensor.detach().cpu().tolist()]

    def _leg_debug_stats(self) -> dict[str, torch.Tensor]:
        raw_actions = self.env.unwrapped.action_manager.action
        action_term = self.env.unwrapped.action_manager.get_term("vmc")
        actions = getattr(action_term, "processed_actions", raw_actions).clone()
        torques = self._robot.data.applied_torque
        dof_vel = self._robot.data.joint_vel
        dof_pos = self._robot.data.joint_pos

        left_leg_torque = torques[:, self._leg_joint_ids[:2]]
        right_leg_torque = torques[:, self._leg_joint_ids[2:4]]
        wheel_torque = torques[:, self._wheel_joint_ids]
        wheel_vel = dof_vel[:, self._wheel_joint_ids]
        vmc_wheel_vel = wheel_vel

        # Base velocity (in body frame) vs command
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        commands = self.env.unwrapped.command_manager.get_command("base_velocity")
        command_ranges = self.env.unwrapped.command_manager.get_term("base_velocity").cfg.ranges
        projected_gravity = self._robot.data.projected_gravity_b

        # VMC task-space state
        from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state
        vmc_cfg = self.env.unwrapped.cfg.vmc_actions
        clip_tp_actions = float(getattr(vmc_cfg, "clip_tp_actions", getattr(vmc_cfg, "clip_actions", 100.0)))
        clip_force_actions = float(getattr(vmc_cfg, "clip_force_actions", getattr(vmc_cfg, "clip_actions", 100.0)))
        clip_wheel_actions = float(getattr(vmc_cfg, "clip_wheel_actions", getattr(vmc_cfg, "clip_actions", 100.0)))
        actions[:, [0, 3]].clamp_(-clip_tp_actions, clip_tp_actions)
        actions[:, [1, 4]].clamp_(-clip_force_actions, clip_force_actions)
        actions[:, [2, 5]].clamp_(-clip_wheel_actions, clip_wheel_actions)
        if float(getattr(vmc_cfg, "action_scale_vel", 0.0)) == 0.0 or clip_wheel_actions <= 0.0:
            actions[:, [2, 5]] = 0.0
        left_action = actions[:, :3]
        right_action = actions[:, 3:6]
        vmc_state = compute_vmc_state(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            leg_joint_indices=list(self._leg_joint_ids),
            wheel_joint_indices=list(self._wheel_joint_ids),
            l1=vmc_cfg.l1,
            l2=vmc_cfg.l2,
            offset=vmc_cfg.offset,
            theta1_offset=vmc_cfg.theta1_offset,
            theta2_offset=vmc_cfg.theta2_offset,
        )

        # Task-space force/torque from actions
        tp_cmd = torch.stack([actions[:, 0], actions[:, 3]], dim=1) * vmc_cfg.action_scale_tp
        delta_force_cmd = torch.stack([actions[:, 1], actions[:, 4]], dim=1) * vmc_cfg.action_scale_force
        total_force_cmd = delta_force_cmd + vmc_cfg.feedforward_force
        wheel_vel_ref = torch.stack([actions[:, 2], actions[:, 5]], dim=1) * vmc_cfg.action_scale_vel
        wheel_vel_error = wheel_vel_ref - vmc_wheel_vel
        if hasattr(self._robot.data, "torque_limit"):
            wheel_torque_limit = self._robot.data.torque_limit[:, self._wheel_joint_ids].mean(dim=0)
        else:
            wheel_torque_limit = torch.tensor(
                self.env.unwrapped.cfg.actions.vmc.torque_limits[-2:],
                device=wheel_torque.device,
                dtype=wheel_torque.dtype,
            )
        wheel_speed_radius = 0.058
        if hasattr(self._robot.data, "soft_joint_vel_limits"):
            wheel_joint_vel_limit = self._robot.data.soft_joint_vel_limits[:, self._wheel_joint_ids]
        else:
            wheel_joint_vel_limit = torch.zeros_like(wheel_vel)
        if hasattr(self._robot.data, "torque_limit"):
            wheel_asset_torque_limit = self._robot.data.torque_limit[:, self._wheel_joint_ids]
        else:
            wheel_asset_torque_limit = wheel_torque_limit.unsqueeze(0).expand_as(wheel_torque)

        step_dt = float(getattr(self.env.unwrapped, "step_dt", 0.02))
        if self._prev_debug_commands is None or self._prev_debug_commands.shape != commands.shape:
            self._prev_debug_commands = commands.detach().clone()
            self._prev_debug_base_lin_vel = base_lin_vel.detach().clone()
            self._prev_debug_base_ang_vel = base_ang_vel.detach().clone()
            self._steps_since_command_change = torch.full(
                (commands.shape[0],), 1_000_000, device=commands.device, dtype=torch.long
            )
        command_delta = commands - self._prev_debug_commands
        command_changed = command_delta[:, [0, 2]].abs().amax(dim=1) > 1.0e-3
        self._steps_since_command_change = torch.where(
            command_changed,
            torch.zeros_like(self._steps_since_command_change),
            self._steps_since_command_change + 1,
        )
        response_window_steps = max(1, int(round(0.5 / max(step_dt, 1.0e-6))))
        response_mask = self._steps_since_command_change <= response_window_steps
        base_lin_acc = (base_lin_vel - self._prev_debug_base_lin_vel) / max(step_dt, 1.0e-6)
        base_ang_acc = (base_ang_vel - self._prev_debug_base_ang_vel) / max(step_dt, 1.0e-6)
        lin_x_error_signed = commands[:, 0] - base_lin_vel[:, 0]
        ang_z_error_signed = commands[:, 2] - base_ang_vel[:, 2]
        lin_x_error = lin_x_error_signed.abs()
        ang_z_error = ang_z_error_signed.abs()
        if response_mask.any():
            response_lin_x_error = lin_x_error[response_mask].mean()
            response_ang_z_error = ang_z_error[response_mask].mean()
            response_env_frac = response_mask.float().mean()
        else:
            response_lin_x_error = torch.zeros((), device=commands.device, dtype=commands.dtype)
            response_ang_z_error = torch.zeros((), device=commands.device, dtype=commands.dtype)
            response_env_frac = torch.zeros((), device=commands.device, dtype=commands.dtype)
        self._prev_debug_commands = commands.detach().clone()
        self._prev_debug_base_lin_vel = base_lin_vel.detach().clone()
        self._prev_debug_base_ang_vel = base_ang_vel.detach().clone()

        tp_actions = torch.stack([actions[:, 0], actions[:, 3]], dim=1)
        force_actions = torch.stack([actions[:, 1], actions[:, 4]], dim=1)
        leg_actions = torch.cat([tp_actions, force_actions], dim=1)
        wheel_actions = torch.stack([actions[:, 2], actions[:, 5]], dim=1)
        tp_sat_threshold = 0.95 * max(clip_tp_actions, 1.0e-6)
        force_sat_threshold = 0.95 * max(clip_force_actions, 1.0e-6)
        wheel_sat_threshold = 0.95 * max(clip_wheel_actions, 1.0e-6)
        if clip_wheel_actions <= 0.0:
            wheel_sat_frac = torch.zeros((), device=actions.device, dtype=actions.dtype)
        else:
            wheel_sat_frac = wheel_actions.abs().gt(wheel_sat_threshold).float().mean()

        return {
            "left_action_mean_abs": left_action.abs().mean(dim=0),
            "right_action_mean_abs": right_action.abs().mean(dim=0),
            "action_abs_mean": actions.abs().mean(),
            "action_abs_max": actions.abs().max(),
            "raw_action_abs_mean": raw_actions.abs().mean(),
            "raw_action_abs_max": raw_actions.abs().max(),
            "action_sat_frac_0p95": torch.cat(
                [
                    tp_actions.abs() / max(clip_tp_actions, 1.0e-6),
                    force_actions.abs() / max(clip_force_actions, 1.0e-6),
                    wheel_actions.abs() / max(clip_wheel_actions, 1.0e-6),
                ],
                dim=1,
            ).gt(0.95).float().mean(),
            "leg_action_abs_mean": leg_actions.abs().mean(),
            "tp_action_sat_frac_0p95": tp_actions.abs().gt(tp_sat_threshold).float().mean(),
            "force_action_sat_frac_0p95": force_actions.abs().gt(force_sat_threshold).float().mean(),
            "wheel_action_abs_mean": wheel_actions.abs().mean(),
            "wheel_action_sat_frac_0p95": wheel_sat_frac,
            "clip_tp_actions": torch.tensor(clip_tp_actions, device=actions.device),
            "clip_force_actions": torch.tensor(clip_force_actions, device=actions.device),
            "clip_wheel_actions": torch.tensor(clip_wheel_actions, device=actions.device),
            "left_torque_mean_abs": left_leg_torque.abs().mean(dim=0),
            "right_torque_mean_abs": right_leg_torque.abs().mean(dim=0),
            "wheel_torque_mean_abs": wheel_torque.abs().mean(dim=0),
            "wheel_torque_abs_mean": wheel_torque.abs().mean(),
            "wheel_torque_abs_max": wheel_torque.abs().max(),
            "wheel_torque_limit": wheel_torque_limit,
            "wheel_torque_sat_frac_0p95": (wheel_torque.abs() > 0.95 * wheel_torque_limit).float().mean(),
            "wheel_asset_torque_limit_mean": wheel_asset_torque_limit.mean(dim=0),
            "wheel_vel_mean_abs": vmc_wheel_vel.abs().mean(dim=0),
            "wheel_vel_abs_mean": vmc_wheel_vel.abs().mean(),
            "wheel_vel_abs_max": vmc_wheel_vel.abs().max(),
            "wheel_vel_ref_abs_mean": wheel_vel_ref.abs().mean(),
            "wheel_vel_ref_abs_max": wheel_vel_ref.abs().max(),
            "wheel_vel_error_abs_mean": wheel_vel_error.abs().mean(),
            "wheel_vel_error_abs_max": wheel_vel_error.abs().max(),
            "wheel_joint_vel_limit_mean": wheel_joint_vel_limit.mean(dim=0),
            "wheel_ground_speed_ref_abs_mean": wheel_vel_ref.abs().mean() * wheel_speed_radius,
            "wheel_ground_speed_ref_abs_max": wheel_vel_ref.abs().max() * wheel_speed_radius,
            "wheel_ground_speed_abs_mean": vmc_wheel_vel.abs().mean() * wheel_speed_radius,
            "wheel_ground_speed_abs_max": vmc_wheel_vel.abs().max() * wheel_speed_radius,
            "left_action_env0": left_action[0],
            "right_action_env0": right_action[0],
            "left_torque_env0": left_leg_torque[0],
            "right_torque_env0": right_leg_torque[0],
            "wheel_torque_env0": wheel_torque[0],
            "wheel_vel_env0": vmc_wheel_vel[0],
            "base_lin_vel_mean": base_lin_vel.mean(dim=0),
            "base_ang_vel_mean": base_ang_vel.mean(dim=0),
            "commands_mean": commands.mean(dim=0),
            "command_lin_x_abs_mean": commands[:, 0].abs().mean(),
            "command_lin_x_abs_max": commands[:, 0].abs().max(),
            "command_ang_z_abs_mean": commands[:, 2].abs().mean(),
            "command_ang_z_abs_max": commands[:, 2].abs().max(),
            "command_delta_lin_x_abs_mean": command_delta[:, 0].abs().mean(),
            "command_delta_lin_x_abs_max": command_delta[:, 0].abs().max(),
            "command_delta_ang_z_abs_mean": command_delta[:, 2].abs().mean(),
            "command_delta_ang_z_abs_max": command_delta[:, 2].abs().max(),
            "command_lin_x_range": torch.tensor(command_ranges.lin_vel_x, device=commands.device, dtype=commands.dtype),
            "command_ang_z_range": torch.tensor(command_ranges.ang_vel_z, device=commands.device, dtype=commands.dtype),
            "lin_x_error_abs_mean": lin_x_error.mean(),
            "lin_x_error_abs_max": lin_x_error.max(),
            "lin_x_error_signed_mean": lin_x_error_signed.mean(),
            "ang_z_error_abs_mean": ang_z_error.mean(),
            "ang_z_error_abs_max": ang_z_error.max(),
            "ang_z_error_signed_mean": ang_z_error_signed.mean(),
            "base_lin_x_acc_abs_mean": base_lin_acc[:, 0].abs().mean(),
            "base_lin_x_acc_abs_max": base_lin_acc[:, 0].abs().max(),
            "base_ang_z_acc_abs_mean": base_ang_acc[:, 2].abs().mean(),
            "base_ang_z_acc_abs_max": base_ang_acc[:, 2].abs().max(),
            "response_0p5s_env_frac": response_env_frac,
            "response_0p5s_lin_x_error_abs_mean": response_lin_x_error,
            "response_0p5s_ang_z_error_abs_mean": response_ang_z_error,
            "base_lin_x_mean": base_lin_vel[:, 0].mean(),
            "base_lin_x_abs_mean": base_lin_vel[:, 0].abs().mean(),
            "base_ang_z_mean": base_ang_vel[:, 2].mean(),
            "base_ang_z_abs_mean": base_ang_vel[:, 2].abs().mean(),
            "projected_gravity_mean": projected_gravity.mean(dim=0),
            "tilt_xy_abs_mean": projected_gravity[:, :2].norm(dim=1).mean(),
            "tilt_xy_abs_max": projected_gravity[:, :2].norm(dim=1).max(),
            "upright_factor_mean": (torch.clamp(-projected_gravity[:, 2], 0, 0.7) / 0.7).mean(),
            "base_lin_vel_env0": base_lin_vel[0],
            "base_ang_vel_env0": base_ang_vel[0],
            "commands_env0": commands[0],
            "theta0_mean": vmc_state["theta0"].mean(dim=0),
            "theta0_lr_error_abs_mean": (vmc_state["theta0"][:, 0] - vmc_state["theta0"][:, 1]).abs().mean(),
            "L0_mean": vmc_state["L0"].mean(dim=0),
            "L0_lr_error_abs_mean": (vmc_state["L0"][:, 0] - vmc_state["L0"][:, 1]).abs().mean(),
            "tp_cmd_mean": tp_cmd.mean(dim=0),
            "delta_force_cmd_mean": delta_force_cmd.mean(dim=0),
            "total_force_cmd_mean": total_force_cmd.mean(dim=0),
            "wheel_vel_ref_mean": wheel_vel_ref.abs().mean(dim=0),
            "theta0_env0": vmc_state["theta0"][0],
            "L0_env0": vmc_state["L0"][0],
            "tp_cmd_env0": tp_cmd[0],
            "delta_force_cmd_env0": delta_force_cmd[0],
            "total_force_cmd_env0": total_force_cmd[0],
            "joint_wheel_vel_env0": wheel_vel[0],
            "wheel_vel_ref_env0": wheel_vel_ref[0],
            "wheel_radius": torch.tensor(wheel_speed_radius, device=actions.device),
            "action_scale_vel": torch.tensor(float(vmc_cfg.action_scale_vel), device=actions.device),
            "max_wheel_ground_speed_from_action": torch.tensor(
                float(vmc_cfg.action_scale_vel) * wheel_speed_radius, device=actions.device
            ),
            "max_wheel_ground_speed_from_clip": torch.tensor(
                float(vmc_cfg.action_scale_vel) * wheel_speed_radius * clip_wheel_actions, device=actions.device
            ),
        }

    def _reward_debug_stats(self) -> dict[str, float]:
        reward_manager = getattr(self.env.unwrapped, "reward_manager", None)
        if reward_manager is None or not hasattr(reward_manager, "_episode_sums"):
            return {}

        stats = {}
        max_episode_length_s = float(getattr(self.env.unwrapped, "max_episode_length_s", 1.0))
        for name, value in reward_manager._episode_sums.items():
            if isinstance(value, torch.Tensor):
                stats[name] = float((value / max(max_episode_length_s, 1e-6)).detach().mean().cpu().item())
        return stats

    def _write_diagnostics_jsonl(
        self,
        it: int,
        leg_stats: dict[str, torch.Tensor],
        reward_stats: dict[str, float],
        rewbuffer,
        lenbuffer,
        mean_value_loss,
        mean_surrogate_loss,
        mean_kl,
        mean_extra_loss,
        fps: int,
    ):
        if self._diagnostics_path is None:
            return
        row = {
            "iteration": int(it),
            "total_timesteps": int(self.tot_timesteps),
            "fps": int(fps),
            "mean_reward": statistics.mean(rewbuffer) if len(rewbuffer) > 0 else None,
            "mean_episode_length": statistics.mean(lenbuffer) if len(lenbuffer) > 0 else None,
            "loss": {
                "value_function": float(mean_value_loss),
                "surrogate": float(mean_surrogate_loss),
                "kl": float(mean_kl),
                "encoder": float(mean_extra_loss),
            },
            "policy": {
                "mean_noise_std": self._as_float(self.alg.actor_critic.std.mean()),
                "action_abs_mean": self._as_float(leg_stats["action_abs_mean"]),
                "action_abs_max": self._as_float(leg_stats["action_abs_max"]),
                "raw_action_abs_mean": self._as_float(leg_stats["raw_action_abs_mean"]),
                "raw_action_abs_max": self._as_float(leg_stats["raw_action_abs_max"]),
                "action_sat_frac_0p95": self._as_float(leg_stats["action_sat_frac_0p95"]),
                "leg_action_abs_mean": self._as_float(leg_stats["leg_action_abs_mean"]),
                "tp_action_sat_frac_0p95": self._as_float(leg_stats["tp_action_sat_frac_0p95"]),
                "force_action_sat_frac_0p95": self._as_float(leg_stats["force_action_sat_frac_0p95"]),
                "wheel_action_abs_mean": self._as_float(leg_stats["wheel_action_abs_mean"]),
                "wheel_action_sat_frac_0p95": self._as_float(leg_stats["wheel_action_sat_frac_0p95"]),
                "clip_tp_actions": self._as_float(leg_stats["clip_tp_actions"]),
                "clip_force_actions": self._as_float(leg_stats["clip_force_actions"]),
                "clip_wheel_actions": self._as_float(leg_stats["clip_wheel_actions"]),
            },
            "command_tracking": {
                "command_lin_x_abs_mean": self._as_float(leg_stats["command_lin_x_abs_mean"]),
                "command_lin_x_abs_max": self._as_float(leg_stats["command_lin_x_abs_max"]),
                "command_delta_lin_x_abs_mean": self._as_float(leg_stats["command_delta_lin_x_abs_mean"]),
                "command_delta_lin_x_abs_max": self._as_float(leg_stats["command_delta_lin_x_abs_max"]),
                "command_lin_x_range": self._as_list(leg_stats["command_lin_x_range"]),
                "actual_lin_x_mean": self._as_float(leg_stats["base_lin_x_mean"]),
                "actual_lin_x_abs_mean": self._as_float(leg_stats["base_lin_x_abs_mean"]),
                "actual_lin_x_acc_abs_mean": self._as_float(leg_stats["base_lin_x_acc_abs_mean"]),
                "actual_lin_x_acc_abs_max": self._as_float(leg_stats["base_lin_x_acc_abs_max"]),
                "lin_x_error_abs_mean": self._as_float(leg_stats["lin_x_error_abs_mean"]),
                "lin_x_error_abs_max": self._as_float(leg_stats["lin_x_error_abs_max"]),
                "lin_x_error_signed_mean": self._as_float(leg_stats["lin_x_error_signed_mean"]),
                "command_ang_z_abs_mean": self._as_float(leg_stats["command_ang_z_abs_mean"]),
                "command_ang_z_abs_max": self._as_float(leg_stats["command_ang_z_abs_max"]),
                "command_delta_ang_z_abs_mean": self._as_float(leg_stats["command_delta_ang_z_abs_mean"]),
                "command_delta_ang_z_abs_max": self._as_float(leg_stats["command_delta_ang_z_abs_max"]),
                "command_ang_z_range": self._as_list(leg_stats["command_ang_z_range"]),
                "actual_ang_z_mean": self._as_float(leg_stats["base_ang_z_mean"]),
                "actual_ang_z_abs_mean": self._as_float(leg_stats["base_ang_z_abs_mean"]),
                "actual_ang_z_acc_abs_mean": self._as_float(leg_stats["base_ang_z_acc_abs_mean"]),
                "actual_ang_z_acc_abs_max": self._as_float(leg_stats["base_ang_z_acc_abs_max"]),
                "ang_z_error_abs_mean": self._as_float(leg_stats["ang_z_error_abs_mean"]),
                "ang_z_error_abs_max": self._as_float(leg_stats["ang_z_error_abs_max"]),
                "ang_z_error_signed_mean": self._as_float(leg_stats["ang_z_error_signed_mean"]),
                "response_0p5s_env_frac": self._as_float(leg_stats["response_0p5s_env_frac"]),
                "response_0p5s_lin_x_error_abs_mean": self._as_float(
                    leg_stats["response_0p5s_lin_x_error_abs_mean"]
                ),
                "response_0p5s_ang_z_error_abs_mean": self._as_float(
                    leg_stats["response_0p5s_ang_z_error_abs_mean"]
                ),
            },
            "wheels": {
                "radius_m": self._as_float(leg_stats["wheel_radius"]),
                "action_scale_vel_rad_s": self._as_float(leg_stats["action_scale_vel"]),
                "max_ground_speed_from_action_m_s": self._as_float(leg_stats["max_wheel_ground_speed_from_action"]),
                "max_ground_speed_from_clip_m_s": self._as_float(leg_stats["max_wheel_ground_speed_from_clip"]),
                "vel_abs_mean_rad_s": self._as_float(leg_stats["wheel_vel_abs_mean"]),
                "vel_abs_max_rad_s": self._as_float(leg_stats["wheel_vel_abs_max"]),
                "vel_ref_abs_mean_rad_s": self._as_float(leg_stats["wheel_vel_ref_abs_mean"]),
                "vel_ref_abs_max_rad_s": self._as_float(leg_stats["wheel_vel_ref_abs_max"]),
                "vel_error_abs_mean_rad_s": self._as_float(leg_stats["wheel_vel_error_abs_mean"]),
                "vel_error_abs_max_rad_s": self._as_float(leg_stats["wheel_vel_error_abs_max"]),
                "ground_speed_abs_mean_m_s": self._as_float(leg_stats["wheel_ground_speed_abs_mean"]),
                "ground_speed_abs_max_m_s": self._as_float(leg_stats["wheel_ground_speed_abs_max"]),
                "ground_speed_ref_abs_mean_m_s": self._as_float(leg_stats["wheel_ground_speed_ref_abs_mean"]),
                "ground_speed_ref_abs_max_m_s": self._as_float(leg_stats["wheel_ground_speed_ref_abs_max"]),
                "torque_abs_mean_nm": self._as_float(leg_stats["wheel_torque_abs_mean"]),
                "torque_abs_max_nm": self._as_float(leg_stats["wheel_torque_abs_max"]),
                "torque_sat_frac_0p95": self._as_float(leg_stats["wheel_torque_sat_frac_0p95"]),
                "torque_limit_nm_lr": self._as_list(leg_stats["wheel_torque_limit"]),
                "asset_torque_limit_nm_lr_mean": self._as_list(leg_stats["wheel_asset_torque_limit_mean"]),
                "joint_vel_limit_rad_s_lr_mean": self._as_list(leg_stats["wheel_joint_vel_limit_mean"]),
            },
            "posture": {
                "tilt_xy_abs_mean": self._as_float(leg_stats["tilt_xy_abs_mean"]),
                "tilt_xy_abs_max": self._as_float(leg_stats["tilt_xy_abs_max"]),
                "upright_factor_mean": self._as_float(leg_stats["upright_factor_mean"]),
                "projected_gravity_mean": self._as_list(leg_stats["projected_gravity_mean"]),
                "theta0_lr_error_abs_mean": self._as_float(leg_stats["theta0_lr_error_abs_mean"]),
                "L0_lr_error_abs_mean": self._as_float(leg_stats["L0_lr_error_abs_mean"]),
                "tp_cmd_mean_nm": self._as_list(leg_stats["tp_cmd_mean"]),
                "delta_force_cmd_mean_n": self._as_list(leg_stats["delta_force_cmd_mean"]),
                "total_force_cmd_mean_n": self._as_list(leg_stats["total_force_cmd_mean"]),
            },
            "env0": {
                "command": self._as_list(leg_stats["commands_env0"]),
                "base_lin_vel": self._as_list(leg_stats["base_lin_vel_env0"]),
                "base_ang_vel": self._as_list(leg_stats["base_ang_vel_env0"]),
                "left_action": self._as_list(leg_stats["left_action_env0"]),
                "right_action": self._as_list(leg_stats["right_action_env0"]),
                "wheel_vel": self._as_list(leg_stats["wheel_vel_env0"]),
                "wheel_vel_ref": self._as_list(leg_stats["wheel_vel_ref_env0"]),
                "wheel_torque": self._as_list(leg_stats["wheel_torque_env0"]),
                "theta0": self._as_list(leg_stats["theta0_env0"]),
                "L0": self._as_list(leg_stats["L0_env0"]),
                "tp_cmd": self._as_list(leg_stats["tp_cmd_env0"]),
                "delta_force_cmd": self._as_list(leg_stats["delta_force_cmd_env0"]),
                "total_force_cmd": self._as_list(leg_stats["total_force_cmd_env0"]),
            },
            "rewards_per_second": reward_stats,
        }
        os.makedirs(os.path.dirname(self._diagnostics_path), exist_ok=True)
        with open(self._diagnostics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    def _log_diagnostics_to_tensorboard(
        self, it: int, leg_stats: dict[str, torch.Tensor], reward_stats: dict[str, float]
    ):
        scalars = {
            "Diagnostics/action_abs_mean": leg_stats["action_abs_mean"],
            "Diagnostics/action_abs_max": leg_stats["action_abs_max"],
            "Diagnostics/raw_action_abs_mean": leg_stats["raw_action_abs_mean"],
            "Diagnostics/raw_action_abs_max": leg_stats["raw_action_abs_max"],
            "Diagnostics/action_sat_frac_0p95": leg_stats["action_sat_frac_0p95"],
            "Diagnostics/leg_action_abs_mean": leg_stats["leg_action_abs_mean"],
            "Diagnostics/tp_action_sat_frac_0p95": leg_stats["tp_action_sat_frac_0p95"],
            "Diagnostics/force_action_sat_frac_0p95": leg_stats["force_action_sat_frac_0p95"],
            "Diagnostics/wheel_action_abs_mean": leg_stats["wheel_action_abs_mean"],
            "Diagnostics/wheel_action_sat_frac_0p95": leg_stats["wheel_action_sat_frac_0p95"],
            "Diagnostics/clip_tp_actions": leg_stats["clip_tp_actions"],
            "Diagnostics/clip_force_actions": leg_stats["clip_force_actions"],
            "Diagnostics/clip_wheel_actions": leg_stats["clip_wheel_actions"],
            "Diagnostics/lin_x_error_abs_mean": leg_stats["lin_x_error_abs_mean"],
            "Diagnostics/lin_x_error_abs_max": leg_stats["lin_x_error_abs_max"],
            "Diagnostics/lin_x_error_signed_mean": leg_stats["lin_x_error_signed_mean"],
            "Diagnostics/command_delta_lin_x_abs_mean": leg_stats["command_delta_lin_x_abs_mean"],
            "Diagnostics/command_delta_lin_x_abs_max": leg_stats["command_delta_lin_x_abs_max"],
            "Diagnostics/base_lin_x_acc_abs_mean": leg_stats["base_lin_x_acc_abs_mean"],
            "Diagnostics/base_lin_x_acc_abs_max": leg_stats["base_lin_x_acc_abs_max"],
            "Diagnostics/response_0p5s_lin_x_error_abs_mean": leg_stats["response_0p5s_lin_x_error_abs_mean"],
            "Diagnostics/command_lin_x_range_max": leg_stats["command_lin_x_range"][1],
            "Diagnostics/ang_z_error_abs_mean": leg_stats["ang_z_error_abs_mean"],
            "Diagnostics/ang_z_error_abs_max": leg_stats["ang_z_error_abs_max"],
            "Diagnostics/ang_z_error_signed_mean": leg_stats["ang_z_error_signed_mean"],
            "Diagnostics/command_delta_ang_z_abs_mean": leg_stats["command_delta_ang_z_abs_mean"],
            "Diagnostics/command_delta_ang_z_abs_max": leg_stats["command_delta_ang_z_abs_max"],
            "Diagnostics/base_ang_z_acc_abs_mean": leg_stats["base_ang_z_acc_abs_mean"],
            "Diagnostics/base_ang_z_acc_abs_max": leg_stats["base_ang_z_acc_abs_max"],
            "Diagnostics/response_0p5s_ang_z_error_abs_mean": leg_stats["response_0p5s_ang_z_error_abs_mean"],
            "Diagnostics/response_0p5s_env_frac": leg_stats["response_0p5s_env_frac"],
            "Diagnostics/command_ang_z_range_max": leg_stats["command_ang_z_range"][1],
            "Diagnostics/wheel_vel_abs_mean_rad_s": leg_stats["wheel_vel_abs_mean"],
            "Diagnostics/wheel_vel_abs_max_rad_s": leg_stats["wheel_vel_abs_max"],
            "Diagnostics/wheel_vel_ref_abs_mean_rad_s": leg_stats["wheel_vel_ref_abs_mean"],
            "Diagnostics/wheel_vel_ref_abs_max_rad_s": leg_stats["wheel_vel_ref_abs_max"],
            "Diagnostics/wheel_vel_error_abs_mean_rad_s": leg_stats["wheel_vel_error_abs_mean"],
            "Diagnostics/wheel_vel_error_abs_max_rad_s": leg_stats["wheel_vel_error_abs_max"],
            "Diagnostics/wheel_ground_speed_abs_mean_m_s": leg_stats["wheel_ground_speed_abs_mean"],
            "Diagnostics/wheel_ground_speed_abs_max_m_s": leg_stats["wheel_ground_speed_abs_max"],
            "Diagnostics/wheel_ground_speed_ref_abs_mean_m_s": leg_stats["wheel_ground_speed_ref_abs_mean"],
            "Diagnostics/wheel_ground_speed_ref_abs_max_m_s": leg_stats["wheel_ground_speed_ref_abs_max"],
            "Diagnostics/wheel_torque_abs_mean_nm": leg_stats["wheel_torque_abs_mean"],
            "Diagnostics/wheel_torque_abs_max_nm": leg_stats["wheel_torque_abs_max"],
            "Diagnostics/wheel_torque_sat_frac_0p95": leg_stats["wheel_torque_sat_frac_0p95"],
            "Diagnostics/tilt_xy_abs_mean": leg_stats["tilt_xy_abs_mean"],
            "Diagnostics/tilt_xy_abs_max": leg_stats["tilt_xy_abs_max"],
            "Diagnostics/upright_factor_mean": leg_stats["upright_factor_mean"],
            "Diagnostics/theta0_lr_error_abs_mean": leg_stats["theta0_lr_error_abs_mean"],
            "Diagnostics/L0_lr_error_abs_mean": leg_stats["L0_lr_error_abs_mean"],
            "Diagnostics/tp_cmd_abs_mean_nm": leg_stats["tp_cmd_mean"].abs().mean(),
            "Diagnostics/delta_force_cmd_abs_mean_n": leg_stats["delta_force_cmd_mean"].abs().mean(),
            "Diagnostics/total_force_cmd_mean_n": leg_stats["total_force_cmd_mean"].mean(),
            "Diagnostics/action_scale_vel_rad_s": leg_stats["action_scale_vel"],
            "Diagnostics/max_wheel_ground_speed_from_action_m_s": leg_stats["max_wheel_ground_speed_from_action"],
            "Diagnostics/max_wheel_ground_speed_from_clip_m_s": leg_stats["max_wheel_ground_speed_from_clip"],
        }
        for name, value in scalars.items():
            self.writer.add_scalar(name, self._as_float(value), it)
        for name, value in reward_stats.items():
            self.writer.add_scalar(f"RewardDebug/{name}", value, it)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        print("[DEBUG][WlSequenceRunner] learn start")
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_td = self.env.get_observations()
        print("[DEBUG][WlSequenceRunner] initial observations for learn fetched")
        obs, obs_history, critic_obs = self._split_obs(obs_td)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        total_it = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, total_it):
            print(f"[DEBUG][WlSequenceRunner] rollout iteration {it} start")
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, critic_obs)
                    next_obs_td, rewards, dones, infos = self.env.step(actions)
                    next_obs, next_history, next_critic = self._split_obs(next_obs_td)
                    self.alg.process_env_step(rewards.to(self.device), dones.to(self.device), infos, next_obs)
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards.to(self.device)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    obs, obs_history, critic_obs = next_obs, next_history, next_critic

                collect_time = time.time() - start
                start = time.time()
                critic_obs_input = torch.cat((critic_obs, self.alg.actor_critic.encode(obs_history)), dim=-1)
                self.alg.compute_returns(critic_obs_input)

            mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss = self.alg.update()
            learn_time = time.time() - start
            if self.log_dir is not None:
                self._log(it, num_learning_iterations, collect_time, learn_time, rewbuffer, lenbuffer, ep_infos,
                          mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss)
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            self.current_learning_iteration = it

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log(
        self,
        it,
        total_iterations,
        collect_time,
        learn_time,
        rewbuffer,
        lenbuffer,
        ep_infos,
        mean_value_loss,
        mean_surrogate_loss,
        mean_kl,
        mean_extra_loss,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collect_time + learn_time
        iteration_time = collect_time + learn_time
        fps = int(self.num_steps_per_env * self.env.num_envs / max(collect_time + learn_time, 1e-6))
        self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
        self.writer.add_scalar("Loss/encoder", mean_extra_loss, it)
        self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
        self.writer.add_scalar("Policy/mean_noise_std", self.alg.actor_critic.std.mean().item(), it)
        self.writer.add_scalar("Policy/mean_kl", mean_kl, it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        if len(rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
        for ep_info in ep_infos[:1]:
            for key, value in ep_info.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], device=self.device)
                self.writer.add_scalar(f"Episode/{key}", value.float().mean(), it)

        reward_text = "n/a"
        length_text = "n/a"
        if len(rewbuffer) > 0:
            reward_text = f"{statistics.mean(rewbuffer):.2f}"
            length_text = f"{statistics.mean(lenbuffer):.2f}"
        eta_seconds = 0.0
        if it >= 0:
            eta_seconds = self.tot_time / (it + 1) * max(total_iterations - it - 1, 0)
        leg_stats = self._leg_debug_stats()
        reward_stats = self._reward_debug_stats()
        self._log_diagnostics_to_tensorboard(it, leg_stats, reward_stats)
        self._write_diagnostics_jsonl(
            it,
            leg_stats,
            reward_stats,
            rewbuffer,
            lenbuffer,
            mean_value_loss,
            mean_surrogate_loss,
            mean_kl,
            mean_extra_loss,
            fps,
        )
        width = 92
        title = f" Learning iteration {it}/{total_iterations - 1} "
        print("#" * width)
        print(title.center(width, " "))
        print()
        print(f"{'Total steps:':>34} {self.tot_timesteps}")
        print(f"{'Steps per second:':>34} {fps}")
        print(f"{'Collection time:':>34} {collect_time:.3f}s")
        print(f"{'Learning time:':>34} {learn_time:.3f}s")
        print(f"{'Iteration time:':>34} {iteration_time:.2f}s")
        print(f"{'Mean value loss:':>34} {mean_value_loss:.4f}")
        print(f"{'Mean surrogate loss:':>34} {mean_surrogate_loss:.4f}")
        print(f"{'Mean KL:':>34} {mean_kl:.4f}")
        print(f"{'Mean encoder loss:':>34} {mean_extra_loss:.4f}")
        print(f"{'Mean reward:':>34} {reward_text}")
        print(f"{'Mean episode length:':>34} {length_text}")
        print(f"{'Mean action std:':>34} {self.alg.actor_critic.std.mean().item():.2f}")
        print(f"{'Time elapsed:':>34} {self.tot_time:.1f}s")
        print(f"{'ETA:':>34} {eta_seconds:.1f}s")
        print("-" * width)
        print(f"{'Left action |mean abs| [Tp, dF, wheel]:':>34} {self._format_tensor(leg_stats['left_action_mean_abs'])}")
        print(f"{'Right action |mean abs| [Tp, dF, wheel]:':>34} {self._format_tensor(leg_stats['right_action_mean_abs'])}")
        print(f"{'Env0 left action [Tp, dF, wheel]:':>34} {self._format_tensor(leg_stats['left_action_env0'])}")
        print(f"{'Env0 right action [Tp, dF, wheel]:':>34} {self._format_tensor(leg_stats['right_action_env0'])}")
        print(f"{'Left leg torque |mean abs| [hip, knee]:':>34} {self._format_tensor(leg_stats['left_torque_mean_abs'])}")
        print(f"{'Right leg torque |mean abs| [hip, knee]:':>34} {self._format_tensor(leg_stats['right_torque_mean_abs'])}")
        print(f"{'Env0 left leg torque [hip, knee]:':>34} {self._format_tensor(leg_stats['left_torque_env0'])}")
        print(f"{'Env0 right leg torque [hip, knee]:':>34} {self._format_tensor(leg_stats['right_torque_env0'])}")
        print(f"{'Wheel torque |mean abs| [L, R]:':>34} {self._format_tensor(leg_stats['wheel_torque_mean_abs'])}")
        print(f"{'Env0 wheel torque [L, R]:':>34} {self._format_tensor(leg_stats['wheel_torque_env0'])}")
        print(f"{'Wheel vel |mean abs| [L, R]:':>34} {self._format_tensor(leg_stats['wheel_vel_mean_abs'])}")
        print(f"{'Env0 wheel vel [L, R]:':>34} {self._format_tensor(leg_stats['wheel_vel_env0'])}")
        print(f"{'Env0 wheel vel ref [L, R]:':>34} {self._format_tensor(leg_stats['wheel_vel_ref_env0'])}")
        print(f"{'theta0 |mean| [L, R]:':>34} {self._format_tensor(leg_stats['theta0_mean'])}")
        print(f"{'L0 |mean| [L, R]:':>34} {self._format_tensor(leg_stats['L0_mean'])}")
        print(f"{'Tp cmd |mean| [L, R] Nm:':>34} {self._format_tensor(leg_stats['tp_cmd_mean'])}")
        print(f"{'deltaF cmd |mean| [L, R] N:':>34} {self._format_tensor(leg_stats['delta_force_cmd_mean'])}")
        print(f"{'total F cmd |mean| [L, R] N:':>34} {self._format_tensor(leg_stats['total_force_cmd_mean'])}")
        print(f"{'Env0 theta0 [L, R]:':>34} {self._format_tensor(leg_stats['theta0_env0'])}")
        print(f"{'Env0 L0 [L, R]:':>34} {self._format_tensor(leg_stats['L0_env0'])}")
        print(f"{'Env0 Tp cmd [L, R] Nm:':>34} {self._format_tensor(leg_stats['tp_cmd_env0'])}")
        print(f"{'Env0 deltaF cmd [L, R] N:':>34} {self._format_tensor(leg_stats['delta_force_cmd_env0'])}")
        print(f"{'Env0 total F cmd [L, R] N:':>34} {self._format_tensor(leg_stats['total_force_cmd_env0'])}")
        print(f"{'Base lin vel |mean| [x,y,z]:':>34} {self._format_tensor(leg_stats['base_lin_vel_mean'])}")
        print(f"{'Commands |mean| [x,ang,head]:':>34} {self._format_tensor(leg_stats['commands_mean'])}")
        print(f"{'Env0 base lin vel [x,y,z]:':>34} {self._format_tensor(leg_stats['base_lin_vel_env0'])}")
        print(f"{'Env0 commands [x,ang,head]:':>34} {self._format_tensor(leg_stats['commands_env0'])}")
        print("#" * width)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "extra_optimizer_state_dict": self.alg.extra_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_optimizer and "extra_optimizer_state_dict" in loaded_dict:
            self.alg.extra_optimizer.load_state_dict(loaded_dict["extra_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is None:
            device = self.device
        self.alg.actor_critic.to(device)

        def policy(obs_td):
            actor_obs = obs_td.get(self.actor_group).to(device)
            history_obs = obs_td.get(self.history_group).to(device)
            actions, _ = self.alg.actor_critic.act_inference(actor_obs, history_obs)
            return actions

        return policy
