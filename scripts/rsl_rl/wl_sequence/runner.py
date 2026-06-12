from __future__ import annotations

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
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.git_status_repos = []
        self._robot = self.env.unwrapped.scene["robot"]
        self._leg_joint_ids, _ = self._robot.find_joints(["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"], preserve_order=True)
        self._wheel_joint_ids, _ = self._robot.find_joints(["l_wheel_Joint", "r_wheel_Joint"], preserve_order=True)

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

    def _leg_debug_stats(self) -> dict[str, torch.Tensor]:
        actions = self.env.unwrapped.action_manager.action
        torques = self._robot.data.applied_torque
        dof_vel = self._robot.data.joint_vel
        dof_pos = self._robot.data.joint_pos

        left_action = actions[:, :3]
        right_action = actions[:, 3:6]
        left_leg_torque = torques[:, self._leg_joint_ids[:2]]
        right_leg_torque = torques[:, self._leg_joint_ids[2:4]]
        wheel_torque = torques[:, self._wheel_joint_ids]
        wheel_vel = dof_vel[:, self._wheel_joint_ids]

        # Base velocity (in body frame) vs command
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        commands = self.env.unwrapped.command_manager.get_command("base_velocity")

        # VMC task-space state
        from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state
        vmc_cfg = self.env.unwrapped.cfg.vmc_actions
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

        # Task-space reference from actions
        theta0_ref = torch.stack([actions[:, 0], actions[:, 3]], dim=1) * vmc_cfg.action_scale_theta + vmc_cfg.theta0_offset
        l0_ref = torch.stack([actions[:, 1], actions[:, 4]], dim=1) * vmc_cfg.action_scale_l0 + vmc_cfg.l0_offset
        wheel_vel_ref = torch.stack([actions[:, 2], actions[:, 5]], dim=1) * vmc_cfg.action_scale_vel

        return {
            "left_action_mean_abs": left_action.abs().mean(dim=0),
            "right_action_mean_abs": right_action.abs().mean(dim=0),
            "left_torque_mean_abs": left_leg_torque.abs().mean(dim=0),
            "right_torque_mean_abs": right_leg_torque.abs().mean(dim=0),
            "wheel_torque_mean_abs": wheel_torque.abs().mean(dim=0),
            "wheel_vel_mean_abs": wheel_vel.abs().mean(dim=0),
            "left_action_env0": left_action[0],
            "right_action_env0": right_action[0],
            "left_torque_env0": left_leg_torque[0],
            "right_torque_env0": right_leg_torque[0],
            "wheel_torque_env0": wheel_torque[0],
            "wheel_vel_env0": wheel_vel[0],
            "base_lin_vel_mean": base_lin_vel.mean(dim=0),
            "base_ang_vel_mean": base_ang_vel.mean(dim=0),
            "commands_mean": commands.mean(dim=0),
            "base_lin_vel_env0": base_lin_vel[0],
            "commands_env0": commands[0],
            "theta0_mean": vmc_state["theta0"].mean(dim=0),
            "theta0_ref_mean": theta0_ref.mean(dim=0),
            "L0_mean": vmc_state["L0"].mean(dim=0),
            "L0_ref_mean": l0_ref.mean(dim=0),
            "wheel_vel_ref_mean": wheel_vel_ref.abs().mean(dim=0),
            "theta0_env0": vmc_state["theta0"][0],
            "theta0_ref_env0": theta0_ref[0],
            "L0_env0": vmc_state["L0"][0],
            "L0_ref_env0": l0_ref[0],
            "wheel_vel_env0": wheel_vel[0],
            "wheel_vel_ref_env0": wheel_vel_ref[0],
        }

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
        print(f"{'Left action |mean abs| [theta, L0, wheel]:':>34} {self._format_tensor(leg_stats['left_action_mean_abs'])}")
        print(f"{'Right action |mean abs| [theta, L0, wheel]:':>34} {self._format_tensor(leg_stats['right_action_mean_abs'])}")
        print(f"{'Env0 left action [theta, L0, wheel]:':>34} {self._format_tensor(leg_stats['left_action_env0'])}")
        print(f"{'Env0 right action [theta, L0, wheel]:':>34} {self._format_tensor(leg_stats['right_action_env0'])}")
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
        print(f"{'theta0 ref |mean| [L, R]:':>34} {self._format_tensor(leg_stats['theta0_ref_mean'])}")
        print(f"{'L0 |mean| [L, R]:':>34} {self._format_tensor(leg_stats['L0_mean'])}")
        print(f"{'L0 ref |mean| [L, R]:':>34} {self._format_tensor(leg_stats['L0_ref_mean'])}")
        print(f"{'Env0 theta0 [L, R]:':>34} {self._format_tensor(leg_stats['theta0_env0'])}")
        print(f"{'Env0 theta0 ref [L, R]:':>34} {self._format_tensor(leg_stats['theta0_ref_env0'])}")
        print(f"{'Env0 L0 [L, R]:':>34} {self._format_tensor(leg_stats['L0_env0'])}")
        print(f"{'Env0 L0 ref [L, R]:':>34} {self._format_tensor(leg_stats['L0_ref_env0'])}")
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
