import datetime as dt
import logging
import sys
import warnings
from collections import deque
from enum import StrEnum
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from typer import Typer

from RLroutes_PAPER.utils.conf import load_conf
from RLroutes_PAPER.utils.drift import (
    PositionError,
    filter_buoy_nd_date,
    pretty_series_print,
)
from RLroutes_PAPER.utils.travel_time import (
    compute_haversine_distance,
    great_circle_travel_time,
)

MAX_LAT = 90
MAX_LON = 180
DAY_PERIODICITY = 365
HOUR_PERIODICITY = 24

Action = int
Reward = float
State = pd.Series
State_norm = np.ndarray
Environment = pd.Series

logging.basicConfig(level=logging.INFO)

# a large number of DEBUG warnings appear on the log due to matplotlib
logging.getLogger("matplotlib.font_manager").disabled = True


class Repeat_buoys(StrEnum):
    """Whether to allow to repeat buoy visit in OceanWithBuoys episodes.

    Attributes
    ----------
    ALLOW : str
        "allow", allow to repeat buoys during the episode.
    ZERO_TONS : str
        "zero_tons", set to zero the captured tons when repeating visited buoy.
    _INF_TONS : str
        "_inf_tons", set to -INF the captured tons when repeating visited buoy.
    NO_STAY : str
        "no_stay", mask the actions corresponding to the current buoy.
    PROHIBITED : str
        "prohibited", mask the actions corresponding to repeated buoys.
    """

    ALLOW = "allow"
    ZERO_TONS = "zero_tons"
    _INF_TONS = "_inf_tons"
    NO_STAY = "no_stay"
    PROHIBITED = "prohibited"


class Termination(StrEnum):
    """Types of episode termination condition.

    Attributes
    ----------
    MAX_CAPTURE : str
        "max_capture", terminate the episode when a capture is reached.
    LATLON : str
        "latlon", terminate the episode when an specific latlon (within a
        tolerance) is reached.
    """

    MAX_CAPTURE = "max_capture"
    LATLON = "latlon"


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


app = Typer()


class StatesMemory:
    """Store stack of states.

    Useful to capture state sequential information such as temporal dependence.

    References
    ----------
    .. [1] https://docs.python.org/3/library/collections.html#collections.deque
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state: State | State_norm):
        """Save state / normalized state.

        Save normalized state is recommended.

        When `capacity` is reached, `deque` iterator automatically remove older
        elements when new ones are appended. [1]
        """
        self.memory.append(state)

    def __len__(self):
        """Memory len."""
        return len(self.memory)


def buoys_pos_and_tons(
    datetime: dt.datetime,
    buoy_keys: np.typing.ArrayLike,
    df_buoys: pd.DataFrame,
    synthetic_catch_delta: float,
    timedelta: dt.timedelta = dt.timedelta(hours=1),
    rng_or_seed: np.random.Generator | int | None = None,
    buoy_col: str = "KEY_BALIZA",
    datetime_col: str = "FECHA_HORA",
    lat_col: str = "LATITUD",
    lon_col: str = "LONGITUD",
    capture_col: str | None = "CAPTURA",
) -> pd.Series:
    """Retrieve position and tons of input buoys based on input date and time.

    Parameters
    ----------
    df_buoys : pd.DataFrame
        Buoys along time information.
    datetime : dt.datetime
        Date and time to retrieve buoy pos and tons.
    buoy_keys : np.typing.ArrayLike
        Strings of the different buoys to travel to.
    synthetic_catch_delta : float
        Perform synthetic predictions of catch adding/subtracting a random value
        within delta to catches in `df_buoys`.
    timedelta: dt.timedelta
        Delta of time to accept retrieved data, by default one hour,
        dt.timedelta(hours=1).
    rng_or_seed: np.random.Generator | int | None, optional
        [Only necessary if `synthetic_catch_delta`]. Random generator or seed to
        use for synthetic predictions. If int or None, obtain random values from
        a newly initialized random generator with `np.random.default_rng()`.
        By default, None.
    buoy_col : str, optional
        Name of the storing column of buoy key, by default "KEY_BALIZA".
    datetime_col : str, optional
        Name of the storing column of datetime, by default "FECHA_HORA".
    lat_col : str, optional
        Name of the storing column of latitudes, by default "LATITUD".
    lon_col : str, optional
        Name of the storing column of longitudes, by default "LONGITUD".
    capture_col : str | None, optional
        [Only necessary if `synthetic_catch_delta`]. Name of the storing column
        of captures. By default "CAPTURA".

    Returns
    -------
    pd.Series
        Name, latitude, longitude and capture prediction on each buoy in
        `buoy_keys` at indicated date and time.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/9528421/value-for-epsilon-in-python
    """
    # address latitude and longitude of buoys for selected datetime
    # buoy keys to lower case
    buoy_keys_lower = np.char.lower(np.array(list(buoy_keys), dtype="str"))
    # filter desired datetime and buoy keys of input `df_buoys`
    df_buoys["time_delta"] = abs(datetime - df_buoys[datetime_col])
    df_buoys_data = (
        df_buoys[
            (df_buoys["time_delta"] <= timedelta)
            & (df_buoys[buoy_col].str.lower().isin(buoy_keys_lower))
        ]
        .reset_index()  # preserve index for future operations
        .copy(deep=True)
    )

    # keep only first datetime record of each buoy within allowed `time_delta`
    df_buoys_data = (
        df_buoys_data.sort_values([buoy_col, datetime_col])
        .groupby(buoy_col)
        .first()
        .reset_index()
        .set_index("index")
    )

    # check if we couldn't retrieve data of any buoy in the environment
    remaining_buoys = set(buoy_keys_lower) - set(df_buoys_data[buoy_col].str.lower())
    if len(remaining_buoys) != 0:
        raise ValueError(f"""Not all buoys lat/lon retrieved from `df_buoys`.
            Buoys left: {remaining_buoys}""")

    # address synthetic capture prediction of buoys
    delta = synthetic_catch_delta
    rng = (
        np.random.default_rng(rng_or_seed)
        if isinstance(rng_or_seed, (int | None))
        else rng_or_seed
    )
    df_buoys_data["PREDICTION"] = df_buoys_data[capture_col] + rng.uniform(
        -delta, delta + sys.float_info.epsilon, df_buoys_data.shape[0]
    )  # [1]

    # go from DataFrame to Series
    # making each column record unique by adding the key of the buoy to it
    ser_buoy_data = pd.Series()
    for col in [buoy_col, lat_col, lon_col, "PREDICTION"]:
        for idx in df_buoys_data.T:
            ser_buoy_data[f"{col}_{df_buoys_data.loc[idx, buoy_col]}"] = (
                df_buoys_data.loc[idx, col]
            )

    return ser_buoy_data


def OceanWithBuoys_init(
    start_loc: pd.Series,
    termination: Termination,
    env_global_obs: bool = False,
    seed_synthetic_catch: int | None = None,
    buoy_col: str = "KEY_BALIZA",
    lat_col: str = "LATITUD",
    lon_col: str = "LONGITUD",
    date_col: str = "FECHA",
    time_col: str = "HORA",
    datetime_col: str = "FECHA_HORA",
    capture_col: str = "CAPTURA",
    conf_path: str = "./RLroutes_PAPER/config.toml",
) -> tuple[
    np.typing.ArrayLike,
    Environment,
    pd.DataFrame,
    pd.DataFrame | None,
]:
    """Summarize the initialization of OceanWithBuoys environment.

    Parameters
    ----------
    start_loc : pd.Series
        Pandas series containing start location of the environment.
    termination: Termination
        Define the episode termination condition. Return a dataframe with
        terminal locations only if `Termination.LATLON`.
    env_global_obs: bool, optional
        Work with states with global information, such as the position and
        predicted capture of tuna in all considered buoys.
    seed_synthetic_catch : int | None, optional
        [Only necessary if `env_global_obs`]
        Seed to use for synthetic catch predictions of initial global
        observations. If None, obtain stochastic non replicable predictions.
        By default, None.
    buoy_col : str, optional
        [Only necessary if `env_global_obs`]. Name of the storing column of buoy
        name, by default "KEY_BALIZA".
    lat_col : str, optional
        Name of the storing column of latitudes, by default "LATITUD".
    lon_col : str, optional
        Name of the storing column of longitudes, by default "LONGITUD".
    date_col : str, optional
        [Only necessary if `env_global_obs` is True]. Name of the storing column
        of date, by default "FECHA".
    time_col : str, optional
        [Only necessary if `env_global_obs` is True]. Name of the storing column
        of time, by default "HORA".
    datetime_col : str, optional
        [Only necessary if `env_global_obs`]. Name of the storing column
        of datetime, by default "FECHA_HORA".
    capture_col : str | None, optional
        Name of the storing column of captures. By default "CAPTURA".
    conf_path : str, optional
        Path to config file, by default "./RLroutes_PAPER/config.toml".

    Returns
    -------
    buoykey_array : np.typing.ArrayLike
        OceanWithBuoys `buoykey_array` input array. Strings of the different
        buoys to travel to, obtained from `buoy_track` data.
    start_env : Environment
        Start environment, single buoy or global depending on `env_global_obs`.
    df_buoys : pd.DataFrame
        OceanWithBuoys `buoys_data` input DataFrame. Information of drift buoys.
    df_terminal_latlon : pd.DataFrame | None
        Dataframe with terminal locations. Only if `termination` is
        Termination.LATLON.

    Notes
    -----
    * For further description of previous OceanWithBuoys indicated inputs,
      please, refer to class OceanWithBuoys below.

    Examples
    --------
    * `start_loc` input example:
        start_datetime = dt.datetime(2018, 3, 27, 1, 40, 12)
        start_loc = pd.Series(
            {
                "KEY_BALIZA": None,
                "LATITUD": -9,
                "LONGITUD": -107,
                "FECHA": start_datetime.date(),
                "HORA": start_datetime.time(),
            }
        )
    """
    cfg = load_conf(conf_path)["environment"]
    df = load_buoy_track(cfg["paths_buoy_track_csv"])

    df_terminal_latlon = (
        pd.read_parquet(cfg["path_terminal_latlon"])
        if termination == Termination.LATLON
        else None
    )

    buoykey_array = df[buoy_col].unique()

    # set start_env for the global states
    if env_global_obs:
        map_buoys = buoys_pos_and_tons(
            datetime=dt.datetime.combine(start_loc[date_col], start_loc[time_col]),
            buoy_keys=buoykey_array,
            df_buoys=df,
            synthetic_catch_delta=cfg["synthetic_catch_delta"],
            timedelta=dt.timedelta(hours=1),
            rng_or_seed=seed_synthetic_catch,
            buoy_col=buoy_col,
            datetime_col=datetime_col,
            lat_col=lat_col,
            lon_col=lon_col,
            capture_col=capture_col,
        )
        start_env = pd.concat((start_loc, map_buoys), axis=0)
    else:
        start_env = start_loc

    return (
        buoykey_array,
        start_env,
        # return columns indicated in config, else all columns
        df[cfg["buoy_track"].get("df_cols", df.columns)],
        df_terminal_latlon,
    )


class OceanWithBuoys(gym.Env):  # type: ignore[type-arg]
    """Custom Environment for deployed buoys that follows gym interface.

    References
    ----------
    .. [1] https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
    .. [2] https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    .. [3] https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
    .. [4] https://gymnasium.farama.org/api/spaces/
    .. [5] https://gymnasium.farama.org/api/env/

    .. [6] https://stackoverflow.com/questions/32191029/getting-the-indices-of-several-elements-in-a-numpy-array-at-once

    .. [8] https://github.com/araffin/rl-tutorial-jnrr19?tab=readme-ov-file
    .. [9] https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1

    .. [10] https://www.gymlibrary.dev/content/environment_creation/#reset
    .. [11] https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    """

    # store relevant metadata
    metadata = {"render_fps": 30}

    def __init__(
        self,
        buoykey_array: np.typing.ArrayLike,
        buoys_data: pd.DataFrame,
        start_env: Environment,
        global_obs: bool = False,
        n_states_stack: int | None = None,
        lat_col: str = "LATITUD",
        lon_col: str = "LONGITUD",
        date_col: str = "FECHA",
        time_col: str = "HORA",
        tunapred_col: str = "PREDICTION",
        buoy_col: str = "KEY_BALIZA",
        capture_col: str = "CAPTURA",
        repeat_buoys: Repeat_buoys = Repeat_buoys.ZERO_TONS,
        termination: Termination = Termination.MAX_CAPTURE,
        reward_end_episode: bool = False,
        df_terminal_latlon: pd.DataFrame | None = None,
        conf_path: str = "./RLroutes_PAPER/config.toml",
        inner_conf_path: str = "./RLroutes_PAPER/inner_config.toml",
        env_idx: int | None = None,
    ) -> None:
        """Environment of buoys deployed along the ocean.

        Parameters
        ----------
        buoykey_array: np.typing.ArrayLike
            ArrayLike with strings of the different buoys to travel to.
            It is important to only include those buoys of interest, so we
            restrict the study to the actions of interest.
        buoys_data : pd.DataFrame
            DataFrame with buoys drift.
        start_env : Environment
            Information about the starting point of the environment.
            If buoyKEY do not apply, set their value as None.
        global_obs : bool, optional
            Whether to include information about the whole map to state info.
            Adss more info to the agent but tends to require more memory and
            bigger networks. By default, False.
        n_states_stack: int | None, optional
            Number of states to stack to catch sequential information.
            Adss more info to the agent but tends to require more memory and
            bigger networks. By default, None.
        lat_col : str, optional
            Name of the storing column of latitudes, by default "LATITUD".
        lon_col : str, optional
            Name of the storing column of longitudes, by default "LONGITUD".
        date_col : str, optional
            Name of the storing column of the date, by default "FECHA".
        time_col : str, optional
            Name of the storing column of the time, by default "HOUR".
        tunapred_col : str, optional
            Name of the storing column of tuna capture prediction, by default
            "PREDICTION".
        buoy_col : str, optional
            Name of the storing column of the buoy identifier, by default
            "KEY_BALIZA".
        capture_col : str, optional
            Name of the storing column of tuna capture, by default "CAPTURA".
        repeat_buoys: Repeat_buoys, optional
            Define if buoys can be visited again during the same episode.
            By default, Repeat_buoys.ZERO_TONS
        termination: Termination, optional
            Define the episode termination condition. Thresholds for termination
            (maximum capture / end latitude longitude) and tolerances if apply
            (latitude / longitude tol) defined in config file.
            By default, Termination.MAX_CAPTURE.
        reward_end_episode : bool, optional
            Give a non-zero reward only at the end of episode. By default, False.
        df_terminal_latlon : pd.DataFrame | None, optional
            Latitude/longitude (usually, of port locations) to make a state a
            terminal state. Only necessary if `termination==Termination.LATLON`.
            By default, None.
        conf_path : str, optional
            Path to config file, by default "./RLroutes_PAPER/config.toml".
        inner_conf_path : str, optional
            Path to config file of inner parameters such as predictor parameters, by
            default "./RLroutes_PAPER/inner_config.toml".
        env_idx : int | None, optional
            Index of environment. Useful to keep track of logging of each
            environment when vectorized environments are used. If None, do not
            display any idx. By default, None.

        Warnings
        --------
        * `df_terminal_latlon` only determines the latitude/longitude that, if
          reached, the state is transformed into a terminal state.
          If those locations cannot be reached through the action space, these
          terminal states will never be reached.

        See Also
        --------
        Termination
            Further description of `termination` input.
        """
        super().__init__()
        self.envidx_logging = f"env {env_idx}: " if env_idx is not None else ""

        # load config
        self.conf = load_conf(conf_path)["environment"]
        self.inner_conf = load_conf(inner_conf_path)["environment"]

        # set global obs bool and number of states to stack
        self.global_obs = global_obs
        self.n_states_stack = n_states_stack

        # indicate if reward at the end of episode
        self.reward_end_episode = reward_end_episode

        # define if buoys can be visited more than once per episode
        self._repeat_buoys = repeat_buoys

        # set the termination condition
        self.termination_condition = termination

        if termination == Termination.LATLON:
            if df_terminal_latlon is not None:
                self.terminal_latlon = df_terminal_latlon
            else:
                raise ValueError(
                    "`df_terminal_latlon` must be provided if 'latlon' termination."
                )

        # Store relation between action number and buoy keys, sorted
        # alphabetically and keeping only unique records
        buoys_unique_vals = np.unique(buoykey_array)
        self._action_buoykey_dict = {
            idx: buoykey for idx, buoykey in enumerate(np.sort(buoys_unique_vals))
        }

        # WARNING: `self.state_cols` and `self.global_state_cols` order is VERY
        # VERY VERY important for the remaining pipeline
        self.single_state_cols = [lat_col, lon_col, date_col, time_col]
        self.global_state_cols = np.concatenate(
            [self.single_state_cols]
            + (
                [
                    [
                        f"{lat_col}_{buoy_key}",
                        f"{lon_col}_{buoy_key}",
                        f"{tunapred_col}_{buoy_key}",
                    ]
                    for buoy_key in self._action_buoykey_dict.values()
                ]
                if self.global_obs
                else []
            )
        ).tolist()
        self.env_cols = np.concatenate(
            [self.global_state_cols]
            + [[buoy_col]]
            + (
                [
                    [f"{buoy_col}_{buoy_key}"]
                    for buoy_key in self._action_buoykey_dict.values()
                ]
                if self.global_obs
                else []
            )
        ).tolist()
        self.capture_col = capture_col
        # Store the name of relevant data columns
        self.buoy_col = buoy_col
        self.lat_col, self.lon_col = lat_col, lon_col
        self.date_col, self.time_col = date_col, time_col
        self.datetime_col = f"{date_col}_{time_col}"
        self.tunapred_col = tunapred_col

        # initialize some counters, store current and initial environment and
        # state info, and initialize visited buoys memory
        self._setup(start_env=start_env, mode=Setup_mode.INIT)

        # Order records by time
        self.buoys_data = buoys_data.sort_values([date_col, time_col]).reset_index(
            drop=True
        )

        # ----------- Define action and observation space -----------
        # They must be gym.spaces objects

        # Action space will be the buoy to visit [4]
        self.action_space = gym.spaces.Discrete(np.size(buoys_unique_vals), start=0)

        # State space is composed of 4 elements: latitude, longitude, days and
        # hours (in this order) for current buoy; plus, if `global_obs`
        # selected, latitude, longitude and predicted tuna catch (also in this
        # order) for each one of all buoys

        # We will normalize latitude and longitude to (-1, 1) ranges [1]
        # We will codify days and hours as two different sines with a
        # periodicity of 365 and 24, respectively [1]
        # We also will normalize tuna capture prediction to indicated
        # `max_single_capture`
        lower_bound = (
            [-1] * 4
            + ([-1, -1, 0] * np.size(buoys_unique_vals) if self.global_obs else [])
        ) * (
            self.n_states_stack if self.n_states_stack is not None else 1
        )  # states stack if selected
        higher_bound = (
            [1] * (4 + (3 * np.size(buoys_unique_vals) if self.global_obs else 0))
        ) * (
            self.n_states_stack if self.n_states_stack is not None else 1
        )  # states stack if selected
        self.observation_space = gym.spaces.Box(
            low=np.array(lower_bound), high=np.array(higher_bound), dtype=np.float64
        )  # [4]

    def _setup(self, start_env: Environment, mode: Setup_mode) -> State_norm:
        """Initialize the environment with init and reset methods.

        Initialize some counters, store current and initial environment and
        state info, and initialize visited buoys memory and normalized states memory.

        Parameters
        ----------
        start_env : Environment
            Indicate starting environment.
                * For "INIT" mode, `start_env`
                * For "RESET" mode, `self.init_env`
        mode : Setup_mode
            Select to proceed for `init` or `reset` method.
        """
        # initilaize the counter for the number of transitions of the
        # environment
        self.transition_number = 0

        # initialize the catch, route time and return tracking
        self.caught_tons = 0.0
        self.route_hours = 0.0
        self.rl_return = 0.0
        # Cumulative -inf_capture_value penalties for visiting repeated buoys
        # (Repeat_buoys._INF_TONS). Tracked separately from caught_tons to
        # include in `reward_end_episode` final reward.
        self.inf_penalties = 0.0

        # store and initialize the information about the current environment,
        # relevant for the network training and for storing the step of the
        # environment in a dictionary
        assert set(start_env.index).issubset(
            set(self.env_cols)
        ), f"`start_env` must have {self.env_cols} environment fields."

        # if INIT mode...
        if mode == Setup_mode.INIT:
            # additionally define the initial environment
            self.init_env = start_env
            # and the initial state
            self.init_state = self._normalize_state_values_single(
                start_env[self.single_state_cols]
            )
            if self.global_obs:
                self.init_state = self._normalize_state_values_global(
                    all_buoy_keys=self._action_buoykey_dict.values(),
                    state_global=start_env[self.global_state_cols],
                    norm_single_state=self.init_state,
                    max_capture=self.conf["max_single_capture"],
                )
        # elif RESET mode...
        elif mode == Setup_mode.RESET:
            # check `init_state` is NOT None together with `init_env`
            assert (
                self.init_state is not None
            ), "`init_env` and `init_state` must be no one None."

        # set current environment as start environment
        self.current_env = start_env.copy(deep=True)
        # define norm_current_state for later return
        norm_current_state = self.init_state

        # initialize visited buoys memory
        # CAUTION: initial buoy must be added to memory as it will be always be
        # first visited and, when specified, cannot be selected as an action
        # again
        self._visited_buoys_memory = (
            set()
            if self.current_env[self.buoy_col] is None
            else ({self.action_name_to_idx(self.current_env[self.buoy_col])})
        )

        if self.n_states_stack is not None:
            # initialize normalized states memory (whether we need it or not)
            self._norm_states_memory = StatesMemory(capacity=self.n_states_stack)
            # add first state the same number of times than state memory, providing an
            # static start history
            for _ in range(self.n_states_stack):
                self._norm_states_memory.push(norm_current_state)
            # define `norm_current_state` with current stack of states
            norm_current_state = np.array(self._norm_states_memory.memory).flatten()

        return norm_current_state

    def action_idx_to_name(self, action_idx: int) -> str:
        """Translate action index to the name of the action for user readability."""
        return self._action_buoykey_dict[action_idx]

    def action_name_to_idx(self, action_name: str) -> int:
        """Translate action name to the idx of the action space."""
        return list(self._action_buoykey_dict.values()).index(action_name)

    def step(
        self, action: Action
    ) -> tuple[State_norm, Reward, bool, bool, dict[str, Any]]:
        """Update an environment with actions [5].

        Parameters
        ----------
        action : Action
            Input action representing the buoy to travel to.
            Correspondence between int representation of the action and buoyKEY
            is given by `_action_buoykey_dict` attribute.

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
            Additional information from the environment about the step, i.e.
            debug metrics, capture and journey time info.

        Loggings
        --------
        INFO
            Indicates terminated and truncated events.
        DEBUG
            Indicates selected action, caught tons, journey days, cumulative tons
            caught, step reward, return and more info about locations where the episode
            has terminated.
        """
        info = {}

        # Get buoy key from action
        next_buoy_key = self.action_idx_to_name(action)
        logging.debug(self.envidx_logging + f"Selected action: {next_buoy_key}")

        # ------- NEXT STATE & REWARD -------
        # obtain next state/capture from data
        next_state, journey_hours = self._next_state_from_position(
            df_position=self.buoys_data.rename(
                columns={
                    self.buoy_col: "buoyKEY",
                    self.lon_col: "Longitude",
                    self.lat_col: "Latitude",
                    self.datetime_col: "DateTime",
                }
            ),
            next_buoy_key=next_buoy_key,
        )

        # set capture tons to 0 or -INF if some condition is true
        # WARNING : notice `Repeat_buoys._INF_TONS` condition is addressed
        # first. This is important to not skip it if `self.caught_tons` reached
        # its maximum value `self.conf["max_tons"]`.
        if (
            self._repeat_buoys == Repeat_buoys._INF_TONS
            and action in self._visited_buoys_memory
        ):
            capture_tons = -self.conf["inf_capture_value"]
        elif (
            self.caught_tons >= self.conf["max_tons"]
            or self._repeat_buoys == Repeat_buoys.ZERO_TONS
            and action in self._visited_buoys_memory
        ):
            capture_tons = 0
        # otherwise, estimate it
        else:
            capture_tons = self._retrieve_capture(
                next_buoy_key=next_buoy_key,
                next_state=next_state,
                capture_data=self.buoys_data,
                max_timedelta=dt.timedelta(hours=1),
            )

        if (
            self.conf["reward"]["capture_clip"]
            and capture_tons > self.conf["reward"]["capture_clip"]
        ):
            warnings.warn(
                f"""Captured tons clipped to {self.conf["reward"]["capture_clip"]}
                for {next_buoy_key}, before: {capture_tons}""",
                stacklevel=1,
            )
            capture_tons = self.conf["reward"]["capture_clip"]

        # limit the capture by specified max_tons
        if (self.caught_tons + capture_tons) > self.conf["max_tons"]:
            capture_tons = self.conf["max_tons"] - self.caught_tons

        # update total caught tons if capture reward was not -INF
        if capture_tons != -self.conf["inf_capture_value"]:
            self.caught_tons += capture_tons
            assert capture_tons >= 0
        else:
            self.inf_penalties += capture_tons  # accumulate repeated-buoy penalty
        assert self.caught_tons <= self.conf["max_tons"]
        logging.debug(
            self.envidx_logging + f"Cumulative tons caught: {self.caught_tons}"
        )

        # compute step-by-step in-route hours
        step_hours = journey_hours + self.conf["reward"]["stop_hours"]
        # update accumulated route hours
        self.route_hours += step_hours

        # add one to the count
        self.transition_number += 1

        # check if environment reached the termination condition
        terminated = self._termination(state=next_state[self.single_state_cols])

        # finish the episode if the max number of steps per episode is reached [1]
        truncated = self.transition_number >= self.conf["max_transitions"]

        if truncated:
            logging.info(
                self.envidx_logging
                + "Truncated, maximum number of transitions reached."
            )

        # REWARD: if `reward_end_episode` cumulative reward only provided at the end of
        # the episode
        if self.reward_end_episode:
            reward = (
                self.caught_tons
                + self.inf_penalties  # repeated-buoy penalties (if proceed)
                - self.conf["reward"]["time_weight"] * self.route_hours
                if terminated or truncated
                else 0
            )
        # REWARD: else, compute regular step-by-step reward
        else:
            reward = capture_tons - self.conf["reward"]["time_weight"] * step_hours

        # reward rescale if selected
        reward_rescale = self.conf["reward"]["rescale"]
        reward *= reward_rescale if reward_rescale else 1

        # return track
        self.rl_return += reward

        # add step info
        info["capture_tons"] = capture_tons
        info["journey_hours"] = journey_hours
        info["stop_hours"] = self.conf["reward"]["stop_hours"]

        logging.debug(self.envidx_logging + f"Tons caught: {capture_tons}")
        logging.debug(self.envidx_logging + f"Journey days: {journey_hours / 24}")

        logging.debug(self.envidx_logging + f"Reward: {reward}")
        logging.debug(self.envidx_logging + f"Return: {self.rl_return}")

        # ------- UPDATE VALUES -------
        # reset current env in order to store the new one
        self.current_env = pd.Series(index=self.env_cols, dtype="object")
        # Store next and not normalized environment info in `current_env`
        self.current_env[self.single_state_cols] = next_state[self.single_state_cols]
        self.current_env[self.buoy_col] = next_buoy_key

        if self.global_obs:
            global_buoys = buoys_pos_and_tons(
                datetime=dt.datetime.combine(
                    self.current_env[self.date_col], self.current_env[self.time_col]
                ),
                buoy_keys=self._action_buoykey_dict.values(),
                df_buoys=self.buoys_data,
                synthetic_catch_delta=self.conf["synthetic_catch_delta"],
                timedelta=dt.timedelta(hours=1),
                rng_or_seed=self.np_random,  # [10]
                buoy_col=self.buoy_col,
                datetime_col=self.datetime_col,
                lat_col=self.lat_col,
                lon_col=self.lon_col,
                capture_col=self.capture_col,
            )
            self.current_env[global_buoys.index] = global_buoys

        # normalize each variable of the state
        norm_next_state = self._normalize_state_values_single(
            self.current_env[self.single_state_cols]
        )
        if self.global_obs:
            norm_next_state = self._normalize_state_values_global(
                all_buoy_keys=self._action_buoykey_dict.values(),
                state_global=self.current_env[self.global_state_cols],
                norm_single_state=norm_next_state,
                max_capture=self.conf["max_single_capture"],
            )
        # provide stack of states if selected
        if self.n_states_stack is not None:
            self._norm_states_memory.push(norm_next_state)
            # define `norm_next_state` with current stack of states
            norm_next_state = np.array(self._norm_states_memory.memory).flatten()

        # store visited buoy
        self._visited_buoys_memory.add(action)

        return norm_next_state, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[State_norm, dict[str, Any]]:
        """Reset the environment to an initial state [5].

        Required before calling step and when episode is terminated or
        truncated. This is managed by StableBaselines3, if used [8,9].

        Parameters
        ----------
        seed : int | None, optional
            The seed that is used to initialize the environment’s PRNG
            (np_random).
            * If the environment does not already have a PRNG and seed=None (the
            default option) is passed, a seed will be chosen from some source of
            entropy (e.g. timestamp or /dev/urandom).
            * However, if the environment already has a PRNG and seed=None is
            passed, the PRNG will not be reset.
            * If you pass an integer, the PRNG will be reset even if it already
            exists.
            Usually, you want to pass an integer right after the environment has
            been initialized and then never again. [5]
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
        info = {"Info": "Additional info."}

        # For Custom environments, the first line of reset() should be
        # super().reset(seed=seed) which implements numpy seeding correctly. [5]
        # With it, change seed if selected
        super().reset(seed=seed)

        # initialize some counters, store current and initial environment and
        # state info, and initialize visited buoys memory and stack of states
        norm_current_state = self._setup(start_env=self.init_env, mode=Setup_mode.RESET)

        return norm_current_state, info

    def render(self) -> None:
        """Render the environments.

        To help visualise what the agent see, examples modes are “human”,
        “rgb_array”, “ansi” for text. [5]
        """
        # left for future work
        raise NotImplementedError()

    def close(self) -> None:
        """Close the environment.

        Important when external software is used, i.e. pygame for rendering,
        databases. [5]
        """
        pass

    def action_masks(self) -> list[bool]:
        """Invalid action masking.

        Intended for Maskable PPO implementation of Stable Baselines3 - Contrib.

        Returns
        -------
        list[bool]
            List of bool which indicates valid (True) and invalid (False) actions.
        """
        # retrieve current buoy
        current_buoy = self.current_env[self.buoy_col]

        # set the len of action space
        action_space_len = len(self._action_buoykey_dict)

        # Retrieve the position of the action which corresponds to current buoy
        # and mask the action corresponding to it. Remaining actions are valid
        actions_mask = [True] * action_space_len
        if self._repeat_buoys == Repeat_buoys.NO_STAY and current_buoy is not None:
            action_idx = self.action_name_to_idx(current_buoy)
            actions_mask[action_idx] = False

        # in addition, if selected with `repeat_buoys` env input, avoid
        # previously visited buoys
        elif self._repeat_buoys == Repeat_buoys.PROHIBITED:
            np_actions_mask = np.array(actions_mask)
            np_actions_mask[list(self._visited_buoys_memory)] = False
            actions_mask = np_actions_mask.tolist()

        return actions_mask

    def _next_state_from_position(
        self,
        df_position: pd.DataFrame,
        next_buoy_key: str,
        max_time_delta: dt.timedelta | None = dt.timedelta(days=1),
    ) -> tuple[State, float]:
        """Obtain next state from position data.

        Afterwards, compute desired route for next state's time.

        Parameters
        ----------
        df_position :  pd.DataFrame
            Position data with "buoyKEY", "DateTime", "Latitude" and "Longitude" cols.
        next_buoy_key : str
            Key of the buoy of destination.
        max_time_delta: dt.timedelta | None, optional
            Maximum time_delta to employ recovered position, maximum of one day
            (see `end_date` definition). By default, one day.

        Returns
        -------
        next_state : State
            Next state, given by the values of `self.state_cols`.
        journey_hours : float
            Hours of journey.

        Warnings
        --------
        * It is assumed that next buoy is in the same position that it was when
          the boat started its journey from the previous buoy.
        """
        required_cols = ["buoyKEY", "DateTime", "Latitude", "Longitude"]
        # load positions from data
        # It is assumed that next buoy is in the same position that it was
        # when the boat started its journey from the previous buoy.
        start_date = self.current_env[self.date_col]
        end_date = start_date + dt.timedelta(days=1)

        # filter position
        df_position = filter_buoy_nd_date(
            df_position.copy(deep=True),
            buoy_key=next_buoy_key,
            date_min=start_date,
            date_max=end_date,
            columns=required_cols,
            buoy_nd_date_cols={"buoy_col": "buoyKEY", "date_col": "FECHA"},
        )
        # if no records, throw and error
        if df_position.empty:
            err_msg = f"for {next_buoy_key} KEY, between {start_date}-{end_date}."
            raise PositionError("No position in data" + err_msg)

        # take available hour after and nearest to the hour of the current state
        current_date_nd_hour = pd.Timestamp.combine(
            start_date, self.current_env[self.time_col]
        )
        df_position["time_delta"] = df_position["DateTime"] - current_date_nd_hour
        df_after = df_position[
            df_position["time_delta"].dt.total_seconds() >= 0
        ].sort_values("time_delta", ascending=True, ignore_index=True)
        if df_after.empty:
            raise PositionError(
                f"All positions for {next_buoy_key} are before {current_date_nd_hour}."
            )
        series_position = df_after.iloc[0]

        # make sure we have only one position record left
        assert series_position.shape == (5,)

        # check time_delta of this position not surpass selected max_time_delta
        if series_position["time_delta"] > max_time_delta:
            raise PositionError(
                "Available data has not records within selected `max_time_delta`."
            )

        # obtain the time of the journey along the geodesic
        geodesic_conf = self.inner_conf["geodesic_path"]
        _, geodesic_days, _, _ = great_circle_travel_time(
            lat_start=self.current_env[self.lat_col],
            lon_start=self.current_env[self.lon_col],
            lat_end=series_position["Latitude"],
            lon_end=series_position["Longitude"],
            velocity=geodesic_conf["velocity"],
        )

        if np.isnan(geodesic_days):
            warnings.warn(
                "Next state at the same position of previous state. "
                "Adding a wait of one hour to determine a new state...",
                stacklevel=1,
            )
            geodesic_days = 1 / 24  # one hour

        # finally store all the information of the next state
        next_state_time = current_date_nd_hour + dt.timedelta(days=geodesic_days)
        next_state = pd.Series(
            {
                self.lat_col: series_position["Latitude"],
                self.lon_col: series_position["Longitude"],
                self.date_col: next_state_time.date(),
                self.time_col: next_state_time.time(),
            }
        )
        logging.debug(
            self.envidx_logging
            + f"Transition of env step {self.transition_number} obtained"
            " with geodesic path computation."
        )

        # compute complete journey hours
        journey_hours = geodesic_days * 24

        return next_state, journey_hours

    def _retrieve_capture(
        self,
        next_buoy_key: str,
        next_state: pd.DataFrame,
        capture_data: pd.DataFrame,
        max_timedelta: dt.timedelta = dt.timedelta(hours=1),
    ) -> float:
        """Retrieve capture from `capture_data` for input `next_state`.

        Parameters
        ----------
        next_buoy_key : str
            Key of next state buoy (`KEY_BALIZA` value).
        next_state : pd.DataFrame
            RL next state to estimate the capture for.
        capture_data : pd.DataFrame
            Data with captures and their time.
        max_timedelta : dt.timedelta, optional
            Maximum tine difference to retrieve `next_state` capture from
            `capture_data`, by default dt.timedelta(hours=1).

        Returns
        -------
        capture_tons : float
            Tons of tuna capture.

        Notes
        -----
        * By default, just a difference of an hour is permitted (`max_timedelta`)
          due to the rate of change that can be seen in echosounder data.
        """
        if not {self.datetime_col, self.buoy_col, self.capture_col} <= set(
            capture_data.columns
        ):
            raise ValueError(
                f"`{self.datetime_col}`,`{self.buoy_col}`, "
                f"`{self.capture_col}` cols expected in `capture_data`"
            )
        capture_data = capture_data.copy(deep=True)
        # if there is a capture in `capture_data` for `next_buoy_key` within a
        # difference of `max_timedelta` from `next_state`, take it as the
        # capture
        capture_data["time_delta"] = abs(
            capture_data[self.datetime_col]
            - pd.Timestamp.combine(next_state[self.date_col], next_state[self.time_col])
        )
        # just take records of `next_buoy_key` of interest which do not surpass
        # `max_timedelta`.
        capture_data = capture_data[
            (capture_data[self.buoy_col].str.lower() == next_buoy_key.lower())
            & (capture_data["time_delta"] <= max_timedelta)
        ]
        try:
            # sort and try to extract the closest capture record
            capture_tons = (
                capture_data.sort_values(
                    "time_delta", ascending=True, ignore_index=True
                )[self.capture_col]
                .head(1)
                .item()
            )
        except ValueError as err:
            raise ValueError(
                "Capture data of specified buoy not within selected time delta."
            ) from err
        return 0 if np.isnan(capture_tons) else capture_tons

    def _normalize_state_values_single(self, state: State) -> State_norm:
        """Normalize each one of the state variables (single buoy).

        Parameters
        ----------
        state : State
            Description of the state. Must be given with
            `self.single_state_cols` variables.

        Returns
        -------
        State_norm
            Normalized state values.
        """
        # check expected cols are in input state
        assert set(self.single_state_cols).issubset(
            set(state.index)
        ), f"Expected {self.single_state_cols} in input state."

        # initialize array of storage
        norm_state = np.zeros(shape=4)

        # obtain the order of each variable in the state array [6]
        sorter = np.argsort(self.single_state_cols)
        lat_idx, lon_idx, date_idx, time_idx = sorter[
            np.searchsorted(
                self.single_state_cols,
                [self.lat_col, self.lon_col, self.date_col, self.time_col],
                sorter=sorter,
            )
        ]

        # normalize lat and lon
        norm_state[lat_idx] = state[self.lat_col] / MAX_LAT
        norm_state[lon_idx] = state[self.lon_col] / MAX_LON

        # normalize date
        days_in_year = (
            state[self.date_col] - dt.datetime(state[self.date_col].year, 1, 1).date()
        ).days
        norm_state[date_idx] = np.sin(2 * np.pi * days_in_year / DAY_PERIODICITY)

        # normalize time
        hours_in_day = (
            state[self.time_col].hour
            + state[self.time_col].minute / 60
            + state[self.time_col].second / 3600
        )
        norm_state[time_idx] = np.sin(2 * np.pi * hours_in_day / HOUR_PERIODICITY)

        return norm_state

    def _normalize_state_values_global(
        self,
        all_buoy_keys: np.typing.ArrayLike,
        state_global: State,
        norm_single_state: State_norm,
        max_capture: float,
    ) -> State_norm:
        """Normalize each one of the state variables (global map).

        Parameters
        ----------
        all_buoy_keys : np.ArrayLike
            Names of all buoys in the environment.
        state_global : State
            Description of the state. Must be given with, at least,
            `self.global_state_cols` not contained in `self.single_state_cols`.
        norm_single_state : State_norm
            Normalized state value of the single buoy.
        max_capture : float
            Maximum capture to normalize with respect it.

        Returns
        -------
        State_norm
            Normalized state values with single and global components.
        """
        # check expected cols are in input state
        assert set(self.global_state_cols).issubset(
            set(self.single_state_cols) | set(state_global.index)
        ), f"""Expected {self.global_state_cols} when `state_global` and
        `self.single_state_cols` are paired together."""

        # initialize array of storage
        norm_state = np.concatenate(
            (norm_single_state, np.zeros(shape=(3 * len(all_buoy_keys))))
        )

        # obtain the order of each variable in the state array [6]
        sorter = np.argsort(self.global_state_cols)
        for buoy_key in all_buoy_keys:
            buoy_lat_idx, buoy_lon_idx, buoy_tunapred_idx = sorter[
                np.searchsorted(
                    self.global_state_cols,
                    [
                        f"{self.lat_col}_{buoy_key}",
                        f"{self.lon_col}_{buoy_key}",
                        f"{self.tunapred_col}_{buoy_key}",
                    ],
                    sorter=sorter,
                )
            ]
            # normalize lat and lon
            norm_state[buoy_lat_idx] = (
                state_global[f"{self.lat_col}_{buoy_key}"] / MAX_LAT
            )
            norm_state[buoy_lon_idx] = (
                state_global[f"{self.lon_col}_{buoy_key}"] / MAX_LON
            )
            # normalize predicted capture
            norm_state[buoy_tunapred_idx] = (
                state_global[f"{self.tunapred_col}_{buoy_key}"] / max_capture
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
        # finish the episode if `max_tons` are reached
        if self.termination_condition == Termination.MAX_CAPTURE:
            terminated = self.caught_tons >= self.conf["max_tons"]
            if terminated:
                logging.info(
                    self.envidx_logging
                    + f"Terminated, reached capture of {self.caught_tons}."
                )

        # finish the episode if any terminal location is nearest than `tol_m_to_latlon`
        elif self.termination_condition == Termination.LATLON:
            n_terminal_locs = self.terminal_latlon.shape[0]
            distance_to_latlon = compute_haversine_distance(
                lat1=np.repeat(state[self.lat_col], n_terminal_locs),
                lon1=np.repeat(state[self.lon_col], n_terminal_locs),
                lat2=self.terminal_latlon[self.lat_col],
                lon2=self.terminal_latlon[self.lon_col],
            )
            terminated = (distance_to_latlon < self.conf["tol_m_to_latlon"]).any()

            if terminated:
                logging.debug(self.envidx_logging + "Terminated, terminal location.")

                if logging.root.level <= logging.DEBUG:
                    terminal_locs = [
                        (self.terminal_latlon.iloc[idx], distance_to_loc)
                        for idx, distance_to_loc in enumerate(distance_to_latlon)
                        if distance_to_loc < self.conf["tol_m_to_latlon"]
                    ]
                    for terminal_loc in terminal_locs:
                        logging.debug(
                            self.envidx_logging
                            + f"Distance to {pretty_series_print(terminal_loc[0])}: "
                            f"{terminal_loc[1]} m"
                        )
        else:
            raise NotImplementedError(
                f"{self.termination_condition} termination not implemented."
            )

        return terminated


def load_buoy_track(csv_paths: list[str]) -> pd.DataFrame:
    """Load buoy track information.

    Parameters
    ----------
    csv_paths : list[str]
        List of paths with all csv to load.

    Returns
    -------
    pd.DataFrame
        Loaded and processed data.
    """
    df_list = []
    for csv_path in csv_paths:
        df_list.append(pd.read_csv(csv_path, encoding="latin-1", low_memory=False))

    df = pd.concat(df_list)
    # drop residual column
    df.drop(columns=["Unnamed: 0"], inplace=True)
    # buoy keys to lower case, as expected along the pipeline
    df["KEY_BALIZA"] = df["KEY_BALIZA"].str.lower()

    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%d/%m/%Y %H:%M").dt.date
    df["HORA"] = pd.to_datetime(df["HORA"], format="mixed", dayfirst=True).dt.time
    vect_time_combine = np.vectorize(pd.Timestamp.combine)
    df["FECHA_HORA"] = vect_time_combine(df["FECHA"], df["HORA"])

    return df


@app.command()
def main(
    start_env: Environment | None = None,
    action_str: str | None = None,
    termination: Termination = Termination.MAX_CAPTURE,
    conf_path: str = "./RLroutes_PAPER/config.toml",
) -> None:
    """Test basic features of `OceanWithBuoys` environment.

    Parameters
    ----------
    start_env : Environment | None, optional
        Information about the starting point of the environment. If buoyKEY
        do not apply, set their value as None. By default, None.
    action_str : str | None, optional
        Action to test the step of created agent. If None, select a random
        action of the action space. By default, None.
    termination: Termination, optional
        Define the episode termination condition, by default
        Termination.MAX_CAPTURE.
    conf_path : str, optional
        Path to config file, by default "./RLroutes_PAPER/config.toml".

    See Also
    --------
    OceanWithBuoys.__init__
        Further description of `termination` input.
    Termination
        Further description of `termination` input.
    """
    buoykey_array, start_env, buoys_data, df_terminal_latlon = OceanWithBuoys_init(
        start_loc=start_env, termination=termination, conf_path=conf_path
    )

    env = OceanWithBuoys(
        buoykey_array=buoykey_array,
        buoys_data=buoys_data,
        start_env=start_env,
        termination=termination,
        df_terminal_latlon=df_terminal_latlon,
        conf_path=conf_path,
    )
    # notice reset seed not specified,
    # so a seed will be chosen from some source of entropy
    env.reset()

    norm_next_state = None
    action = (
        env.action_space.sample()
        if action_str is None
        else env.action_name_to_idx(action_str)
    )

    action_try = 0
    max_action_try = 20
    while norm_next_state is None and action_try <= max_action_try:  # type: ignore [redundant-expr]
        try:
            norm_next_state, reward, terminated, truncated, info = env.step(action)
        except PositionError as e:
            action_try += 1
            action = env.action_space.sample()
            warnings.warn(str(e) + "Trying a different action...", stacklevel=1)
    else:
        if action_try > max_action_try:
            warnings.warn(
                "Action with position data could not be found. Run finished.",
                stacklevel=1,
            )
        else:
            print(norm_next_state, reward, terminated, truncated, info)


if __name__ == "__main__":
    app()
