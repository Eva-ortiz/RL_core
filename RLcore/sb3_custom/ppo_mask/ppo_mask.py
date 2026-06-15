import logging
from math import inf  # new
from typing import TypeVar

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.type_aliases import MaybeCallback

from utils import sec_2_day_hour_min  # new

SelfMaskablePPO = TypeVar("SelfMaskablePPO", bound="MaskablePPO")


class MaskablePPO_custom(MaskablePPO):
    """Proximal Policy Optimization (PPO) (clip version) with Invalid Action Masking.

    Minimum changes on the Stable Baselines 3 contrib implementation.
    * Adds a greedy simulation to track current policy.
    * Add logging and logger records.

    Warnings
    --------
    * If all actions are masked, masking will be ignored.

    See Also
    --------
    sb3_contrib.MaskablePPO
        Original implementation of `MaskablePPO`.

    References
    ----------
    .. [1] https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#custom-callback
    """

    def greedy_simulation(self, greedy_env: gym.Env, use_masking: bool = True):
        """Greedy simulation of the current policy.

        Warnings
        --------
        * Intended for non-vectorized environments.
        * The environment of the simulation is always first initialized with the
          agent seed.
        """
        if greedy_env is None:
            raise ValueError("Environment for greedy simulations not provided.")

        # ensure to not change anything from policy
        self.policy.set_training_mode(False)

        # restart the environment exclusive for greedy simulations
        obs, _ = greedy_env.reset(seed=self.seed)

        terminated, truncated = False, False
        rewards = []
        while not terminated and not truncated:
            # This is the only change related to invalid action masking
            # (comment from sb3_contrib.MaskablePPO)
            if use_masking:
                action_masks = get_action_masks(greedy_env)

            action, _ = self.policy.predict(
                observation=obs, deterministic=True, action_masks=action_masks
            )

            action = action.item()
            obs, reward, terminated, truncated, _ = greedy_env.step(action)

            # store rewards
            rewards.append(reward)
        return sum(rewards)

    def learn(  # type: ignore[override]
        self: SelfMaskablePPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        greedy_check_interval: int | None = 1,
        greedy_env: gym.Env | None = None,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskablePPO:
        """Copy of MaskablePPO learn implementation with greedy agent simulation."""
        iteration = 0
        greedy_return = None  # new
        max_greedy_return = -inf  # new

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, self.n_steps, use_masking
            )

            if not continue_training:
                break

            logging.info(f"Step number {self.num_timesteps}/{total_timesteps}")  # new
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                self._dump_logs(iteration)

            self.train()

            # NEW : if policy network updated (train method) and
            # `greedy_check_interval` is not None, perform a greedy simulation
            # of an episode with the current agent to keep track of the most
            # recently obtained policy
            if (
                greedy_check_interval is not None
                and iteration % greedy_check_interval == 0
            ):  # new
                if greedy_env is None:
                    raise ValueError(
                        "greedy_check_interval selected but environment"
                        " for greedy simulations not provided."
                    )  # new
                greedy_return = self.greedy_simulation(greedy_env, use_masking)  # new

                # NEW: if new maximum greedy return, log info with return, timestep
                # and elapsed time
                if greedy_return > max_greedy_return:  # new
                    checkpoint = default_timer()  # new
                    elapsed = checkpoint - learn_start  # new
                    days, hours, minutes = sec_2_day_hour_min(elapsed)  # new
                    logging.info(
                        f"New maximum greedy return {greedy_return} found at "
                        f"{self.num_timesteps} timestep & {days}:{hours}:{minutes:.3f}"
                        " elapsed time in learning process."
                    )  # new
                    max_greedy_return = greedy_return  # new

                self.logger.record("train/greedy_test", greedy_return)  # new
                self.logger.record(
                    "train/greedy_n_experiences", self.num_timesteps
                )  # new
                self.logger.record("train/greedy_n_updates", self._n_updates)  # new

            # NEW : self.num_timesteps corresponds to the total number of steps
            # taken per the number of environments (see self.collect_rollouts
            # and [1])
            self.logger.record("train/n_experiences", self.num_timesteps)  # new

        callback.on_training_end()

        return self
