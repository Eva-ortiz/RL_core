import logging

import gymnasium as gym
import numpy as np


class MixtureEnv(gym.Env):
    """Custom environment that randomly samples between two underlying envs.

    Change of environment is done the beginning of each episode.
    Assumes the two environments have IDENTICAL observation and action spaces.

    Warnings
    --------
    Sub-environments are stored by reference, not copied. Do not share them
    among several MixtureEnv instances living in the same process, as they
    hold the episode state.

    Author
    ------
    Víctor Gallego, Komorebi AI Technologies.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }  # Example metadata

    def __init__(self, env1, env2, mix_prob=0.5, render_mode=None):
        super().__init__()

        self.env1 = env1
        self.env2 = env2

        if self.env1.action_space != self.env2.action_space:
            raise ValueError(
                f"Action spaces must be identical! "
                f"Env1: {self.env1.action_space}, Env2: {self.env2.action_space}"
            )
        if self.env1.observation_space != self.env2.observation_space:
            raise ValueError(
                f"Observation spaces must be identical! "
                f"Env1: {self.env1.observation_space}, "
                f"Env2: {self.env2.observation_space}"
            )

        self.action_space = self.env1.action_space
        self.observation_space = self.env1.observation_space

        self.render_mode = render_mode  # Store for the wrapper's render method

        self.envs = [self.env1, self.env2]

        self.mix_prob = mix_prob  # Probability of choosing env1

        self.active_env = None
        self.active_env_idx = -1

        # check custom attributes of envs are set
        # self.delta_time = (
        #     self.env1.delta_time
        #     if hasattr(self.env1, "delta_time")
        #     else self.env2.delta_time
        #     if hasattr(self.env2, "delta_time")
        #     else None
        # )

    def reset(self, seed=None, options=None):  # noqa: D102
        super().reset(seed=seed)

        # Choose environment for this episode
        if (
            np.random.rand() < self.mix_prob
        ):  # this is not controlled by the seed! But I don't think it's a problem
            self.active_env = self.envs[0]
            self.active_env_idx = 0
        else:
            self.active_env = self.envs[1]
            self.active_env_idx = 1

        logging.debug(f"Active environment index: {self.active_env_idx}")
        # Derive a new seed for the sub-environment to ensure different episode
        # trajectories even if the main seed is the same for multiple resets
        # of MixtureEnv
        sub_seed = None
        if seed is not None:
            # Use the environment's RNG to generate a seed for the sub-environment
            sub_seed = self.np_random.integers(0, np.iinfo(np.int32).max).item()

        obs, info = self.active_env.reset(seed=sub_seed, options=options)

        info["active_env_idx"] = self.active_env_idx
        return obs, info

    def step(self, action):  # noqa: D102
        if self.active_env is None:
            raise RuntimeError("Cannot call step before reset. Call reset() first.")

        obs, reward, terminated, truncated, info = self.active_env.step(action)

        info["active_env_idx"] = self.active_env_idx
        return obs, reward, terminated, truncated, info

    def render(self):  # noqa: D102
        if self.active_env:
            return self.active_env.render()
        return None

    def close(self):  # noqa: D102
        self.env1.close()
        self.env2.close()

    def set_mix_prob(self, prob):  # noqa: D102
        self.mix_prob = np.clip(prob, 0, 1)

    def __getattr__(self, name):
        """Delegate attribute access to active_env if it's set and has the attribute.

        Otherwise, try to delegate to self.env1 (assuming it's representative for
        static attributes like 'action_mapping', 'num_actions' etc., which are
        expected to be consistent across sub-environments).
        This method is only called if the attribute is not found on MixtureEnv itself.
        """
        # Prioritize active_env if it exists and has the attribute
        if self.active_env is not None and hasattr(self.active_env, name):
            return getattr(self.active_env, name)

        # Fallback to env1 for attributes needed before active_env is set
        # (e.g., by wrappers) or if active_env doesn't have it for some reason
        # (though unlikely if envs are similar).
        # This assumes env1 is always initialized and representative.
        if hasattr(self.env1, name):
            return getattr(self.env1, name)

        # If neither active_env nor env1 has it, then it's a genuine AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            "Neither found in sub-environments."
        )


class MixtureEnv3(MixtureEnv):
    """Custom environment that randomly samples between three underlying envs.

    Change of environment is done the beginning of each episode.
    Assumes the three environments have IDENTICAL observation and action spaces.
    """

    def __init__(self, env1, env2, env3, mix_prob=0.33333, render_mode=None):
        super().__init__(env1, env2, render_mode=render_mode)

        self.env3 = env3

        if self.env1.action_space != self.env3.action_space:
            raise ValueError(
                f"Action spaces must be identical! "
                f"Env1: {self.env1.action_space}, Env3: {self.env3.action_space}"
            )
        if self.env1.observation_space != self.env3.observation_space:
            raise ValueError(
                f"Observation spaces must be identical! "
                f"Env1: {self.env1.observation_space}, "
                f"Env3: {self.env3.observation_space}"
            )

        # overwrite self.envs to store three
        self.envs = [self.env1, self.env2, self.env3]

        self.mix_prob_1 = mix_prob  # Threshold under of which choose env1
        self.mix_prob_2 = 2 * mix_prob  # Threshold above of which choose env3

    def reset(self, seed=None, options=None):  # noqa: D102
        gym.Env.reset(self, seed=seed)

        # Choose environment for this episode
        random_env = np.random.rand()
        if (
            random_env < self.mix_prob_1
        ):  # this is not controlled by the seed! But I don't think it's a problem
            self.active_env = self.envs[0]
            self.active_env_idx = 0
        elif random_env > self.mix_prob_2:
            self.active_env = self.envs[2]
            self.active_env_idx = 2
        else:
            self.active_env = self.envs[1]
            self.active_env_idx = 1

        logging.debug(f"Active environment index: {self.active_env_idx}")
        # Derive a new seed for the sub-environment to ensure different episode
        # trajectories even if the main seed is the same for multiple resets
        # of MixtureEnv
        sub_seed = None
        if seed is not None:
            # Use the environment's RNG to generate a seed for the sub-environment
            sub_seed = self.np_random.integers(0, np.iinfo(np.int32).max).item()

        obs, info = self.active_env.reset(seed=sub_seed, options=options)

        info["active_env_idx"] = self.active_env_idx
        return obs, info

    def close(self):  # noqa: D102
        super().close()  # close envs 1 and 2
        self.env3.close()
