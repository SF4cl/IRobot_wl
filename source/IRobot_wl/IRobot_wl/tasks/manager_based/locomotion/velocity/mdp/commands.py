from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_nonzero_uniform(count: torch.Tensor | int, value_range: tuple[float, float], min_abs: float, device: torch.device):
    count_int = int(count.item() if isinstance(count, torch.Tensor) else count)
    if count_int == 0:
        return torch.empty(0, device=device)
    values = torch.empty(count_int, device=device).uniform_(*value_range)
    if min_abs <= 0.0:
        return values
    max_abs = max(abs(value_range[0]), abs(value_range[1]))
    min_abs = min(min_abs, max_abs)
    signs = torch.where(values >= 0.0, 1.0, -1.0)
    zero_signs = torch.where(torch.rand(count_int, device=device) >= 0.5, 1.0, -1.0)
    signs = torch.where(values == 0.0, zero_signs, signs)
    magnitudes = torch.clamp(torch.abs(values), min=min_abs)
    return signs * magnitudes


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class MixedModeVelocityCommand(UniformThresholdVelocityCommand):
    """Velocity command sampler with explicit standing, straight, spin, and arc modes."""

    cfg: "MixedModeVelocityCommandCfg"

    def __init__(self, cfg: "MixedModeVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.base_height_command_b = torch.full(
            (self.num_envs,),
            0.5 * (self.cfg.base_height_range[0] + self.cfg.base_height_range[1]),
            device=self.device,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        count = len(env_ids)
        if count == 0:
            return

        probs = torch.tensor(self.cfg.mode_probabilities, device=self.device, dtype=torch.float)
        probs = probs / torch.sum(probs)
        modes = torch.multinomial(probs, count, replacement=True)

        self.is_standing_env[env_ids] = modes == 0
        self.is_heading_env[env_ids] = False
        self.vel_command_b[env_ids, 1] = 0.0

        stand_mask = modes == 0
        straight_mask = modes == 1
        spin_mask = modes == 2
        arc_mask = modes == 3

        local_ids = torch.as_tensor(env_ids, device=self.device)
        self.vel_command_b[local_ids[stand_mask], :] = 0.0

        self.vel_command_b[local_ids[straight_mask], 0] = _sample_nonzero_uniform(
            straight_mask.sum(), self.cfg.ranges.lin_vel_x, self.cfg.min_lin_vel_x, self.device
        )
        self.vel_command_b[local_ids[straight_mask], 2] = 0.0

        self.vel_command_b[local_ids[spin_mask], 0] = 0.0
        self.vel_command_b[local_ids[spin_mask], 2] = _sample_nonzero_uniform(
            spin_mask.sum(), self.cfg.ranges.ang_vel_z, self.cfg.min_ang_vel_z, self.device
        )

        self.vel_command_b[local_ids[arc_mask], 0] = _sample_nonzero_uniform(
            arc_mask.sum(), self.cfg.ranges.lin_vel_x, self.cfg.min_lin_vel_x, self.device
        )
        self.vel_command_b[local_ids[arc_mask], 2] = _sample_nonzero_uniform(
            arc_mask.sum(), self.cfg.ranges.ang_vel_z, self.cfg.min_ang_vel_z, self.device
        )
        self.base_height_command_b[env_ids] = torch.empty(count, device=self.device).uniform_(
            *self.cfg.base_height_range
        )


@configclass
class MixedModeVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for explicit WL VMC command modes.

    Mode order is: standing, straight, in-place spin, arc turn.
    """

    class_type: type = MixedModeVelocityCommand

    mode_probabilities: tuple[float, float, float, float] = (0.1, 0.3, 0.3, 0.3)
    min_lin_vel_x: float = 0.15
    min_ang_vel_z: float = 0.25
    base_height_range: tuple[float, float] = (0.19, 0.32)


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
