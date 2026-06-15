import logging
from enum import StrEnum
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from typer import Typer
from utils import load_conf

Action = int
Reward = float
State = pd.Series
State_norm = np.ndarray
Environment = pd.Series

logging.basicConfig(level=logging.INFO)

# a large number of DEBUG warnings appear on the log due to matplotlib
logging.getLogger("matplotlib.font_manager").disabled = True

app = Typer()


class Setup_mode(StrEnum):
    """Mode of `setup` method.

    Attributes
    ----------
    INIT : str
        Operate for `init` method.
    RESET : str
        Operate for `reset` method.
    """

    INIT = "init"
    RESET = "reset"


class ENV(gym.Env):  # type: ignore[type-arg]
    """Custom Environment that follows Gymnasium interface.

    References
    ----------
    .. [1] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
    .. [2] https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    .. [3] https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
    .. [4] https://gymnasium.farama.org/api/spaces/
    .. [5] https://gymnasium.farama.org/api/env/

    .. [6] https://github.com/araffin/rl-tutorial-jnrr19?tab=readme-ov-file
    .. [7] https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1

    .. [8] https://stackoverflow.com/questions/32191029/getting-the-indices-of-several-elements-in-a-numpy-array-at-once

    .. [9] https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
    .. [10] https://www.gymlibrary.dev/content/environment_creation/#reset
    .. [11] https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    """

    # relevant metadata storage
    metadata = {"render_fps": 10}

    def __init__(
        self,
        action_names: np.typing.ArrayLike,
        start_env: Environment | Literal["random"] = "random",
        env_idx: int | None = None,
        global_obs: bool = False,
        conf_path: str = "./config.toml",
    ) -> None:
        """Environment.

        Parameters
        ----------
        action_names : np.typing.ArrayLike
            Set of action names that can be performed.
        start_env : Environment | Literal["random"], optional
            Information about the starting point of the environment. If "random"
            instead of pd.Series, pick a random starting point.
            By default, "random".
        env_idx : int | None, optional
            Index of environment. Useful to keep track of logging of each
            environment when vectorized environments are used. If None, do not
            display any idx. By default, None.
        global_obs : bool, optional
            If True, the observation is a global one: besides the features of
            the current state (single observation), it includes features of
            overall environment. If False, only the single observation is used.
            By default, False.
        conf_path : str, optional
            Path to config file, by default "./config.toml".

        Notes
        -----
        * Less or further parameters can be defined here as input; such as the
          terminal condition, database access allowance , etc.
        """
        super().__init__()
        self.envidx_logging = f"env {env_idx}: " if env_idx is not None else ""

        # load config
        self.conf = load_conf(conf_path)["environment"]

        # store whether to use global observations
        self.global_obs = global_obs

        # store relevant names
        self.action_col = "TODO : str"
        self.state_col_1, self.state_col_2 = "TODO : str", "TODO : str"
        self.global_state_col_1, self.global_state_col_2 = (
            "TODO : str",
            "TODO : str",
        )  # if `global_obs`

        # Store relation between action number and its name, sorted
        # alphabetically
        self._action_name_dict = {
            idx: name for idx, name in enumerate(np.sort(action_names))
        }

        # WARNING: `self.single_state_cols` and `self.global_state_cols` order is
        # VERY VERY VERY important for the remaining pipeline

        # single observation: features of the current state
        self.single_state_cols = [self.state_col_1, self.state_col_2]  # [4 EXAMPLE]
        # global features, only used when `global_obs` is True
        self.global_feat_cols = [
            self.global_state_col_1,
            self.global_state_col_2,
        ]  # [4 EXAMPLE]
        # Global observation: single observation + the global features (only
        # when `global_obs` is selected).
        # Here we provide an example where there is a feature associated with
        # each possible action (eg. position, predicted quantity of the
        # reward...)
        self.global_state_cols = np.concatenate(
            [self.single_state_cols]
            + (
                [
                    [
                        f"{self.global_state_col_1}_{action_name}",
                        f"{self.global_state_col_2}_{action_name}",
                    ]  # [4 EXAMPLE]
                    for action_name in self._action_name_dict.values()
                ]
                if self.global_obs
                else []
            )
        ).tolist()
        self.env_cols = self.global_state_cols + [
            "TODO : str",
            "TODO : str",
        ]  # [4 EXAMPLE]

        # initialize some counters, store current and initial environment and
        # state info, and initialize visited actions memory
        self._setup(start_env=start_env, mode=Setup_mode.INIT)

        # ----------- Define action and observation space -----------
        # They must be gym.spaces objects

        # Discrete action space [4 EXAMPLE]
        self.action_space = gym.spaces.Discrete(np.size(action_names), start=0)  # [4]

        # State space composed by continuous values
        # We will usually normalize its values [1]
        # If `global_obs`, the single observation is extended with
        # `len(global_state_cols)` features.
        # Remember we have the example where there is a set of features per
        # possible action, that's the reason for `n_global * n_actions` features.
        n_actions = np.size(action_names)
        n_global_feat = len(self.global_feat_cols)
        lower_bound = [-1] * len(self.single_state_cols) + (
            [-1] * n_global_feat * n_actions if self.global_obs else []
        )  # [4 EXAMPLE]
        higher_bound = [1] * len(self.single_state_cols) + (
            [1] * n_global_feat * n_actions if self.global_obs else []
        )  # [4 EXAMPLE]
        self.observation_space = gym.spaces.Box(
            low=np.array(lower_bound), high=np.array(higher_bound), dtype=np.float64
        )  # [4]

    def action_idx_to_name(self, action_idx: int) -> str:
        """Translate action index to the name of the action for user readability."""
        return self._action_name_dict[action_idx]

    def action_name_to_idx(self, action_name: str) -> int:
        """Translate action name to the idx of the action space."""
        return list(self._action_name_dict.values()).index(action_name)

    def _setup(
        self,
        mode: Setup_mode,
        start_env: Environment | Literal["random"] = None,
        current_env: Environment | None = None,
    ) -> State_norm:
        """Initialize the environment with init and reset methods.

        Initialize some counters, store current and initial environment and
        state info, and initialize visited actions memory.

        Parameters
        ----------
        mode : Setup_mode
            Select to proceed for `init` or `reset` method.
        start_env : Environment | Literal["random"], optional
            Indicate starting environment.
            Only necessary in "INIT" mode, where init env and state are defined.
            If "RESET", current env and state will return to init values.
        current_env : : Environment | None, optional
            Set the environment to arbitrary `current_env`. Useful when we want
            to preserve initial env/state defs, but want to allocate the agent
            into an specific environment state, different from the initial.

        Warnings
        --------
        * We skip the possibility of randomly setting current env/state as a
          terminal state to avoid buggy behaviours.
        """
        # initilaize the counter for the number of transitions of the
        # environment
        self.transition_number = 0

        # initialize the return tracking
        self.rl_return = 0.0

        # store and initialize the information about the current environment,
        # relevant for the network training and for storing the step of the
        # environment in a dictionary
        if mode == Setup_mode.INIT:
            # not valid input
            if start_env is None:
                raise ValueError(
                    "`start_env` must be provided to define initial env and state"
                )
            # random start env
            elif start_env == "random":
                # set both initial state and env as None due to its randomness
                self.init_env, self.init_state = None, None
            # input start env
            else:
                if not set(start_env.index).issubset(set(self.env_cols)):
                    raise ValueError(
                        f"`start_env` must contain {self.env_cols} environment fields."
                    )
                # define the initial environment
                self.init_env = start_env
                # and the initial state
                self.init_state = self._normalize_state_values(start_env)

        # set current environment to that specified, ignoring `init_env` and
        # `init_state` info
        if current_env is not None:
            if not set(current_env.index).issubset(set(self.env_cols)):
                raise ValueError(
                    f"`current_env` must contain {self.env_cols} environment fields."
                )
            self.current_env = current_env
            # compute norm_state for later return
            norm_state = self._normalize_state_values(self.current_env)

        # in randomly initialized environment, `init_state` is None together
        # with `init_env`
        elif self.init_env is None and self.init_state is None:
            # define a random current environment state
            current_env = self._random_env_state()
            while self._termination(current_env[self.single_state_cols]):
                current_env = self._random_env_state()
            self.current_env = current_env
            # compute norm_state for later return
            norm_state = self._normalize_state_values(self.current_env)

        # in static initial env, `init_state` is NOT None together with `init_env`
        elif self.init_env is not None and self.init_state is not None:
            # set current environment as start environment
            self.current_env = self.init_env.copy(deep=True)
            # define norm_state for later return
            norm_state = self.init_state

        else:
            raise AssertionError(
                "Start environment and state are inconsistently defined."
            )

        # initialize visited actions memory
        # CAUTION: initial action must be added to memory as it will be always
        # be first visited and, when specified, cannot be selected as an action
        # again
        self._visited_actions_memory = {
            self.action_name_to_idx(self.current_env[self.action_col])
        }
        return norm_state

    def step(
        self, action: Action
    ) -> tuple[State_norm, Reward, bool, bool, dict[str, Any]]:
        """Update an environment with actions [5].

        Parameters
        ----------
        action : Action
            Input action index.
            Correspondence between idx representation of the action and action
            name is given by `_action_name_dict` attribute.

        Returns
        -------
        norm_next_state : State_norm
            Next agent observation, normalized.
        reward : Reward
            Reward for taking input action.
        terminated : bool
            If the environment has terminated due to input action.
        truncated : bool
            If the environment has truncated due to input action.
        information : dict[str, Any]
            Additional information from the environment about the step.

        Loggings
        --------
        INFO
            Indicates selected action, and terminated and truncated events.
        DEBUG
            Indicates step, reward, return and additional info.
        """
        info = {}

        # Get action name from action idx
        next_action_name = self.action_idx_to_name(action)
        logging.info(self.envidx_logging + f"Selected action: {next_action_name}")

        # ------- NEXT STATE & REWARD -------
        # TODO: Developer must encode here ALL the logic to perform the
        # transition from the state (s) to next state (s') due to input action
        # (a), giving with it the reward value (r)

        # retrieve next state
        next_state = "TODO : State"

        # compute the return
        reward = "TODO : Reward"  # [4 EXAMPLE]
        self.rl_return += reward

        # add relevant info
        info["info_1"] = "TODO : Any"  # [4 EXAMPLE]

        logging.debug(self.envidx_logging + f"Info: {info}")

        logging.debug(self.envidx_logging + f"Reward: {reward}")
        logging.debug(self.envidx_logging + f"Return: {self.rl_return}")

        # ------- UPDATE VALUES -------
        # Store next and not normalized single-observation info in `current_env`
        self.current_env[self.single_state_cols] = next_state[self.single_state_cols]

        # If global observation, recompute and store the global features at the new state.
        # This has to be done as they probably change at every step.
        if self.global_obs:
            self.current_env[self.global_state_cols] = "TODO : State"

        # normalize the observation (single, plus global features if `global_obs`)
        norm_next_state = self._normalize_state_values(self.current_env)

        self.current_env[self.env_cols] = "TODO : Environment"  # [4 EXAMPLE]

        # add termination / truncated conditions
        terminated = "TODO : boolean comparison (e.g.)"  # [4 EXAMPLE]
        if terminated:
            logging.info(
                self.envidx_logging + "Terminated, (To developer: add the reason)."
            )

        # add one to the transition count
        self.transition_number += 1

        # finish the episode if the max number of steps per episode is reached [1]
        truncated = self.transition_number >= self.conf["max_transitions"]

        if truncated:
            logging.info(
                self.envidx_logging
                + "Truncated, maximum number of transitions reached."
            )

        # store visited action [OPTIONAL BEHAVIOR]
        self._visited_actions_memory.add(action)

        return norm_next_state, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State_norm, dict[str, Any]]:
        """Reset the environment to an initial state [5, 11].

        Required before calling step and when episode is terminated or
        truncated. This is managed by StableBaselines3, if used [8,9].

        Parameters
        ----------
        seed : int | None, optional
            [MANDATORY INPUT] The seed that is used to initialize the
            environment's PRNG (np_random).
            * If the environment does not already have a PRNG and seed=None (the
            default option) is passed, a seed will be chosen from some source of
            entropy (e.g. timestamp or /dev/urandom).
            * However, if the environment already has a PRNG and seed=None is
            passed, the PRNG will not be reset.
            * If you pass an integer, the PRNG will be reset even if it already
            exists.
            Usually, you want to pass an integer right after the environment has
            been initialized and then never again. [5, 10]
            By default, None.
        options : dict[str, Any] | None, optional
            Additional information to specify how the environment is reset
            (optional, depending on the specific environment). [5]
            By default, None.

        Returns
        -------
        State_norm
            First agent observation for an episode.
        dict[str, Any]
            Additional information, i.e. metrics, debug info.
        """
        info = {"info_1": "TODO : Any"}  # [4 EXAMPLE]

        # For Custom environments, the first line of reset() should be
        # super().reset(seed=seed) which implements numpy seeding correctly. [5]
        # With it, change seed if selected
        super().reset(seed=seed)

        # initialize some counters, store current and initial environment and
        # state info, and initialize visited actions memory
        norm_current_state = self._setup(mode=Setup_mode.RESET)

        return norm_current_state, info

    def render(self) -> None:
        """Render the environments.

        To help visualise what the agent see, examples modes are “human”,
        “rgb_array”, “ansi” for text. [5]
        """
        raise NotImplementedError()

    def close(self) -> None:
        """Close the environment.

        Important when external software is used, i.e. pygame for rendering,
        databases. [5]
        """
        pass

    def action_masks(self) -> list[bool]:
        """Invalid action masking [OPTIONAL BEHAVIOR].

        Intended for Maskable PPO implementation of Stable Baselines3 - Contrib.
        [9]

        Returns
        -------
        list[bool]
            List of bool which indicates valid (True) and invalid (False) actions.
        """
        # set the len of action space
        action_space_len = len(self._action_name_dict)

        # Mask visited actions. Remaining actions are valid
        # DISCLAIMER : any other masking logic could be implemented, even
        # getting rid of `self._visited_actions_memory`
        np_actions_mask = np.array([True] * action_space_len)  # [4 EXAMPLE]
        np_actions_mask[list(self._visited_actions_memory)] = False  # [4 EXAMPLE]

        return np_actions_mask.tolist()

    def _random_env_state(self) -> Environment:
        """Create a random environment state.

        Returns
        -------
        Environment
            Description of the environment.

        Notes
        -----
        * Random selection of parameters is done through `self.np_random` random
          generator as `reset` initialization is assumed, which first call
          initializes environment's PRNG. [10]
        """
        # generate random numbers
        random_n = self.np_random.random(size=len(self.env_cols))

        # get a random value between min and max vals
        min_val, max_val = "TODO : float", "TODO : float"  # [4 EXAMPLE]
        rand_values = (max_val - min_val) * random_n + min_val  # [4 EXAMPLE]

        env = pd.Series(
            {self.env_cols[idx]: val for idx, val in enumerate(rand_values)}
        )

        # remember: defined state must contain `single_state_cols` (and
        # `global_state_cols` when `global_obs` is selected)
        assert set(self.env_cols) == set(
            env.index
        ), f"Environment must be composed of {self.env_cols} fields."
        return env

    def _normalize_state_values(self, state: State) -> State_norm:
        """Normalize each one of the state variables.

        Parameters
        ----------
        state : State
            State description. Must contain `self.single_state_cols` and 
            also `self.global_state_cols`, if apply.

        Returns
        -------
        State_norm
            Normalized observation values.
        """
        norm_state = self._normalize_state_values_single(state[self.single_state_cols])
        if self.global_obs:
            norm_state = self._normalize_state_values_global(
                all_action_names=self._action_name_dict.values(),
                state_global=state[self.global_state_cols],
                norm_single_state=norm_state
            )
        return norm_state

        Returns
        -------
        State_norm
            Normalized state values.
        """
        # check expected cols are in input state
        assert set(self.state_cols) == (
            set(state.index)
        ), f"Expected {self.state_cols} in input state."

        # initialize array of storage
        norm_state = np.zeros(shape=len(self.state_cols))

        # obtain the order of each variable in the state array [8]
        sorter = np.argsort(self.state_cols)
        var_idx_1, var_idx_2 = sorter[
            np.searchsorted(
                self.state_cols,
                [self.state_col_1, self.state_col_2],
                sorter=sorter,
            )
        ]

        # normalize var1
        max_var_1 = "TODO : float"  # [4 EXAMPLE]
        norm_state[var_idx_1] = state[self.state_col_1] / max_var_1

        # normalize cyclic var2
        periodicity_var_2 = "TODO : float"  # [4 EXAMPLE]
        norm_state[var_idx_2] = np.sin(
            2 * np.pi * state[self.state_col_2] / periodicity_var_2
        )
        return norm_state

    def _termination(self, state: State) -> bool:
        """Return a flag indicating if termination has been reached.

        Parameters
        ----------
        state : State
            State to check if it is a terminal state.

        Returns
        -------
        bool
            True if terminal state, false otherwise.
        """
        # if more than one termination condition [4 EXAMPLE]
        if (
            self.termination_condition == "TODO : StrEnum"
            or self.termination_condition == "TODO : StrEnum"
        ):
            terminated = "TODO : boolean comparison with state input (e.g.)"
        else:
            raise NotImplementedError(
                f"{self.termination_condition} termination not implemented."
            )
        return terminated


@app.command()
def main(
    action_str: str | None = None,
    conf_path: str = "./config.toml",
) -> None:
    """Test basic features of environment.

    Parameters
    ----------
    action_str : str | None, optional
        Action to test the step of created agent. If None, select a random
        action of the action space. By default, None.
    conf_path : str, optional
        Path to config file, by default "./config.toml".
    """
    env = ENV(action_names=["up", "down"], conf_path=conf_path)  # [4 EXAMPLE]
    # notice reset seed not specified,
    # so a seed will be chosen from some source of entropy
    env.reset()

    norm_next_state = None
    action = (
        env.action_space.sample()
        if action_str is None
        else env.action_name_to_idx(action_str)
    )
    norm_next_state, reward, terminated, truncated, info = env.step(action)

    print("Normalized next state:", norm_next_state)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Additional info:", info)


if __name__ == "__main__":
    app()
