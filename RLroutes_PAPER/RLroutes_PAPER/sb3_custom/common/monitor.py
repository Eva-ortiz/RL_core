import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from stable_baselines3.common.monitor import Monitor


class Monitor_custom(Monitor):
    """
    A monitor wrapper for Gym environments.

    It is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is
        done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return
        of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)

    References
    ----------
    .. [1] https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/monitor.html#Monitor
    """

    def __init__(
        self,
        env: gym.Env,
        filename: str | None = None,
        allow_early_resets: bool = True,
        reset_keywords: tuple[str, ...] = (),
        info_keywords: tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        # inherit original __init__ method [1]
        super().__init__(
            env=env,
            filename=filename,
            allow_early_resets=allow_early_resets,
            reset_keywords=reset_keywords,
            info_keywords=info_keywords,
            override_existing=override_existing,
        )

        # additionally store actions, episode returns, episode rewards and
        # episode actions
        self.actions: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_rewards: list[list[float]] = []
        self.episode_actions: list[list[int]] = []

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """
        Call the Gym environment reset.

        Adds functionality to `reset` of SB3 Monitor. [1]

        :param kwargs: Extra keywords saved for the next episode. only if
            defined by reset_keywords
        :return: the first observation of the environment
        """
        # inherit original reset method [1]
        super().reset(**kwargs)

        # additionally reset actions
        self.actions = []

        return self.env.reset(**kwargs)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment with the given action.

        Modified `step` method of [1]. Changes:
        * Redefine `episode_rewards` to `episode_returns`.
        * Add `episode_rewards`. Stores all rewards of all episodes.
        * Add `episode_actions`. Stores all actions of all episodes.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        self.actions.append(action)  # new
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_actions.append(self.actions)  # new
            self.episode_rewards.append(self.rewards)  # new
            self.episode_returns.append(ep_rew)  # modified
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def get_episode_actions(self) -> list[float]:
        """
        Return the actions of all the episodes.

        :return:
        """
        return self.episode_actions

    def get_episode_rewards(self) -> list[float]:
        """
        Return the rewards of all the episodes.

        Modifies `get_episode_rewards` of SB3 Monitor. [1]

        :return:
        """
        return self.episode_rewards

    def get_episode_returns(self) -> list[float]:
        """
        Return the returns of all the episodes.

        Equivalent to `get_episode_rewards` of SB3 Monitor. [1]

        :return:
        """
        return self.episode_returns
