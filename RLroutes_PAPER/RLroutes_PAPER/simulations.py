import ast
import copy
import datetime as dt
import logging
import random
from collections import deque
from itertools import chain, zip_longest
from pathlib import Path
from statistics import mode

import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from RLroutes_PAPER.agent_predict import maskablePPO_episode
from RLroutes_PAPER.agent_train import maskablePPO_train
from RLroutes_PAPER.environment import (
    OceanWithBuoys,
    OceanWithBuoys_init,
    Repeat_buoys,
    Termination,
)
from RLroutes_PAPER.utils.conf import load_conf
from RLroutes_PAPER.utils.drift import timer
from RLroutes_PAPER.utils.plots import Buoys_plot_text, RL_routes_plot
from RLroutes_PAPER.utils.travel_time import great_circle_travel_time

LAT_COL = "LATITUD"
LON_COL = "LONGITUD"
REW_COL = "REWARD"
KEY_COL = "KEY_BALIZA"
DATE_COL = "FECHA"
TIME_COL = "HORA"
DATETIME_COL = "FECHA_HORA"
CAPTURE_COL = "CAPTURA"


def route_random_search(
    timesteps: int, env: gym.Env, seed: int | None = None
) -> tuple[list, list, list, list, list, list, list, list]:
    """Discover the best route from random search in input env.

    Parameters
    ----------
    timesteps : int
        Number of timesteps to execute the random search.
    env : gym.Env
        Initialized Gym environment object where the search will explore.
    seed : int | None, optional
        Seed for environment reset initialization and random module. If None, a
        seed for the environment will be chosen from some source of entropy.

    Returns
    -------
    best_action_sequence : list
        Best episode actions, in order.
    best_reward_sequence : list
        Best episode rewards, in order.
    best_lon_sequence : list
        Best episode longitudes, in order.
    best_lat_sequence : list
        Best episode latitudes, in order.
    best_info_sequence : list
        Best episode steps info, in order.
    return_sequence : list
        Returns obtained for each explored episode, in order.
    episode_step_sequence : list
        Number of the exploration step of the end of each episode, in order.
    n_exp_sequence : list
        Number of experiences of each explored episode, in order.
    """
    info_list = []
    # reset the environment and random to initialize their random number
    if seed is not None:
        random.seed(seed)
    env.reset(seed=seed)

    # initialize best return sequence
    best_action_sequence, best_reward_sequence = [], [-np.inf]
    best_lon_sequence, best_lat_sequence, best_info_sequence = [], [], []
    return_sequence, episode_step_sequence, n_exp_sequence = [], [], []

    # sequence of values to keep track of performed actions
    # start the track from initialized environment
    action_seq = deque([env.current_env[env.buoy_col]])
    reward_seq = deque([0])
    lon_seq = deque([env.current_env[env.lon_col]])
    lat_seq = deque([env.current_env[env.lat_col]])
    n_exp_episode = 0
    for step in range(int(timesteps)):
        # select a random action from those not masked
        not_masked_actions = [
            idx for idx, not_mask in enumerate(env.action_masks()) if not_mask
        ]
        if not not_masked_actions:
            raise RuntimeError(
                f"No available actions at step {step} but episode was not "
                "previously terminated or truncated."
            )
        random_action = random.choice(not_masked_actions)
        logging.debug(f"Available actions at step {step}: {not_masked_actions}")

        # do it in the environment
        _, reward, terminated, truncated, info = env.step(random_action)

        # store episode values
        action_seq.append(env.action_idx_to_name(random_action))
        reward_seq.append(reward)
        lon_seq.append(env.current_env[env.lon_col])
        lat_seq.append(env.current_env[env.lat_col])
        # store output info
        info_list.append(info)
        # add one to number of experiences of current episode
        n_exp_episode += 1

        # restart the environment if terminated or truncated
        if terminated or truncated:
            env.reset()

            # check if last episode was the best and store it if it is
            rl_return = sum(reward_seq)
            if rl_return > sum(best_reward_sequence):
                best_reward_sequence = list(reward_seq)
                best_action_sequence = list(action_seq)
                best_lon_sequence = list(lon_seq)
                best_lat_sequence = list(lat_seq)
                best_info_sequence = info_list
                logging.info(
                    f"Best episode with return {rl_return} obtained in step {step}."
                )

            # save return track
            return_sequence.append(rl_return)
            episode_step_sequence.append(step)
            n_exp_sequence.append(n_exp_episode)

            # reset tracked episode
            action_seq = deque([env.current_env[env.buoy_col]])
            reward_seq = deque([0])
            lon_seq = deque([env.current_env[env.lon_col]])
            lat_seq = deque([env.current_env[env.lat_col]])
            info_list = []
            n_exp_episode = 0

    return (
        best_action_sequence,
        best_reward_sequence,
        best_lon_sequence,
        best_lat_sequence,
        best_info_sequence,
        return_sequence,
        episode_step_sequence,
        n_exp_sequence,
    )


def random_OceanWithBuoys(
    train: bool,
    start_loc: pd.Series,
    termination: Termination = Termination.LATLON,
    repeat_buoys: Repeat_buoys = Repeat_buoys.ZERO_TONS,
    run_seeds: int | list[int | None] | None = None,
    path_out: str = Path("./models/random_oceanwithbuoys"),
    conf_path: str = "./RLroutes_PAPER/config.toml",
):
    """Evaluate the best route from random exploration in OceanWithBuoys env.

    Parameters
    ----------
    train : bool
        Select if we perform the random search (True) or we just analyze
        existent data (False).
    start_loc : pd.Series
        Pandas series containing start info of the boat in the environment.
    termination: Termination, optional
        Define the episode termination condition, by default
        `Termination.LATLON`.
    repeat_buoys: Repeat_buoys, optional
        Define if buoys can be visited again during the same episode.
        By default, Repeat_buoys.ZERO_TONS
    run_seeds : int | list[int | None] | None, optional
        Seed(s) for environment reset initialization and random module of
        each run. If None for seed, a seed for the environment will be chosen
        from some source of entropy.
    path_out : Path
        Path where the plot of the best route will be saved.
    conf_path : str, optional
        Path to config file, by default "./RLroutes_PAPER/config.toml".

    See Also
    --------
    route_random_search : algorithm to simulate the random search.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/55472648/calculate-mean-of-multiple-lists-with-different-lengths
    """
    # create output folder if it does not exist
    Path(path_out).mkdir(parents=True, exist_ok=True)

    if run_seeds is None:
        run_seeds = [None]
    elif isinstance(run_seeds, int):
        run_seeds = [run_seeds]

    # initialize the environment with provided state
    buoykey_array, start_env, buoys_data, df_terminal_latlon = OceanWithBuoys_init(
        start_loc=start_loc,
        termination=termination,
        env_global_obs=False,
        conf_path=conf_path,
    )
    buoys_data_filtered = buoys_data[
        buoys_data[KEY_COL].isin(buoykey_array)
    ]  # filter records of environment buoys for later plots

    if train:
        cfg_random = load_conf(conf_path)["agent"]["random"]
        env = OceanWithBuoys(
            buoykey_array=buoykey_array,
            buoys_data=buoys_data,
            global_obs=False,
            start_env=start_env,
            repeat_buoys=repeat_buoys,
            termination=termination,
            df_terminal_latlon=df_terminal_latlon,
            conf_path=conf_path,
        )
        # ------ ROUTE SEARCH ------
        logging.info("Looking for the best route in OceanWithBuoys environment...")
        with timer(tag="train_time") as train_time:
            results = []
            for seed in run_seeds:
                logging.info(f"\n\tSeed {seed}\n")
                (
                    best_episode_actions,
                    best_episode_rewards,
                    best_episode_lons,
                    best_episode_lats,
                    best_episode_info,
                    return_sequence,
                    step_sequence,
                    n_exp_sequence,
                ) = route_random_search(
                    timesteps=cfg_random["timesteps"], env=env, seed=seed
                )
                results.append(
                    {
                        "best_actions": best_episode_actions,
                        "best_rewards": best_episode_rewards,
                        "best_lons": best_episode_lons,
                        "best_lats": best_episode_lats,
                        "returns": return_sequence,
                        "experiences": n_exp_sequence,
                        "steps": step_sequence,
                        "best_infos": best_episode_info,
                    }
                )  # output relevant info
            df_results = pd.DataFrame(results)
            df_results.to_csv(f"{path_out}/random_train_results_seed_{run_seeds}.csv")
        env.close()
    else:
        df_results = pd.read_csv(
            f"{path_out}/random_train_results_seed_{run_seeds}.csv", index_col=0
        )
        # reset the index to properly differentiate different runs
        df_results.reset_index(drop=True, inplace=True)
        # retrieve columns with complex objects (all for the moment, all are
        # composed of lists)
        for col in df_results.columns:
            df_results[col] = df_results[col].apply(ast.literal_eval)

    # save the most repeated best episode
    best_actions_strs = [
        str(episode_actions) for episode_actions in df_results["best_actions"]
    ]
    most_repeated_actions = mode(best_actions_strs)
    idx_most_repeated_best = best_actions_strs.index(most_repeated_actions)

    # retrieve info of the most repeated best episode
    best_capture_tons, best_journey_hours, best_stop_hours = [], [], []
    for info in df_results.loc[idx_most_repeated_best, "best_infos"]:
        best_capture_tons.append(info["capture_tons"])
        best_journey_hours.append(info["journey_hours"])
        best_stop_hours.append(info["stop_hours"])

    # initialize plot object and arrange plot data
    RL_plot = RL_routes_plot(
        lat_col=LAT_COL,
        lon_col=LON_COL,
        reward_col=REW_COL,
        key_col=KEY_COL,
        save_path=Path(f"{path_out}/episode_seed_{run_seeds}"),
    )
    df_env_buoys = buoys_data_filtered.sort_values(
        by=[f"{DATE_COL}_{TIME_COL}"]
    ).drop_duplicates(
        subset=[KEY_COL], keep="first"
    )  # get first record in time of environment buoys
    df_visited_buoys = pd.DataFrame(
        {
            LAT_COL: df_results.loc[idx_most_repeated_best, "best_lats"],
            LON_COL: df_results.loc[idx_most_repeated_best, "best_lons"],
            REW_COL: df_results.loc[idx_most_repeated_best, "best_rewards"],
            KEY_COL: df_results.loc[idx_most_repeated_best, "best_actions"],
        }
    )
    train_info = {
        "all_buoys": np.unique(buoykey_array).size,
        "n_episodes_per_seed": [
            len(return_seq) for return_seq in df_results["returns"]
        ],
        "n_buoys": len(df_results.loc[idx_most_repeated_best, "best_actions"]),
        "capture": sum(best_capture_tons),
        "t_travel": sum(np.array(best_journey_hours) / 24),
        "t_stop": sum(np.array(best_stop_hours) / 24),
    } | ({"t_train": train_time()} if train else {})

    # Plot the most repeated episode and the disposition of the buoys at its
    # earlier record. Plot it with text with the visit order and buoy key
    RL_plot.plot_buoys(
        df_env_buoys,
        df_visited_buoys,
        text=Buoys_plot_text.VISIT_nd_KEY,
        more_info=train_info,
        title="Random",
        save=True,
    )

    # obtain the mean return per episode during random exploration [1]
    mean_returns = [
        np.nanmean(ep_return)
        for ep_return in zip_longest(*df_results["returns"], fillvalue=np.nan)
    ]
    # plot return sequence during random exploration
    # as a function of episode number
    plt.figure()
    plt.xlabel("Episode")
    plt.ylabel("Mean return (a.u.)")
    plt.plot(mean_returns)
    plt.savefig(f"{path_out}/Mean_return_vs_episode_seed_{run_seeds}.png")

    # as a function of exploration step number (i.e., experiences)
    plt.figure()
    plt.xlabel("Number of experiences")
    plt.ylabel("Returns (a.u.)")
    for idx, return_seq in enumerate(df_results["returns"]):
        # add a common starting point with [0]+
        plt.plot([0] + df_results["steps"][idx], [0] + return_seq)
    plt.legend(run_seeds)
    plt.savefig(f"{path_out}/Returns_vs_experiences_seed_{run_seeds}.png")

    # plot a bar graph of the frequency of each return
    plt.figure()
    # chain all returns from all seeds
    plt.hist(
        list(chain.from_iterable(df_results["returns"])),
        bins=100,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Return values")
    plt.ylabel("Frequency")
    plt.savefig(f"{path_out}/Return_frequency_seed_{run_seeds}.png")

    print("RANDOM SEARCH IS DONE!")


def skipper_OceanWithBuoys(
    start_loc: pd.Series,
    terminal_buoy: str = "b 28",
    terminal_if_negative_score: bool = True,
    path_out: str = Path("./models/skipper_oceanwithbuoys"),
    conf_path: str = "./RLroutes_PAPER/config.toml",
):
    """Evaluate the best route through a greedy myopic exploration.

    More than one visit to the same buoy is not allowed.

    Parameters
    ----------
    start_loc : pd.Series
        Pandas series containing start info of the boat in the environment.
    terminal_buoy: str, optional
        Define the episode terminal buoy, by default "b 28".
    terminal_if_negative_score : bool, optional
        Go to terminal if all scores are negative. By default, True.
    path_out : Path, optional
        Path where the plot of the best route will be saved.
    conf_path : str, optional
        Path to config file, by default "./RLroutes_PAPER/config.toml".

    Notes
    -----
    At each decision step, all non-visited buoys are scored according to the reward
    formula: the estimated biomass at the buoy (including the same uncertainty of the
    biomass used for the biomass input of the state, `synthetic_catch_delta` in config)
    minus the weighted travel time to that buoy.
    The vessel then selects the non-visited buoy with the highest score and repeats this
    process after each visit.
    """
    # create output folder if it does not exist
    Path(path_out).mkdir(parents=True, exist_ok=True)

    # define termination condition of the environment
    termination = Termination.LATLON
    # ------ INIT DATA ------
    # initialize the environment with provided state
    buoykey_array, start_env, buoys_data, df_terminal_latlon = OceanWithBuoys_init(
        start_loc=start_loc,
        termination=termination,
        env_global_obs=True,
        conf_path=conf_path,
    )

    env = OceanWithBuoys(
        buoykey_array=buoykey_array,
        buoys_data=buoys_data,
        global_obs=True,
        start_env=start_env,
        repeat_buoys=Repeat_buoys.PROHIBITED,
        termination=termination,
        df_terminal_latlon=df_terminal_latlon,
        conf_path=conf_path,
    )

    # ------ ROUTE EVALUATION ------
    logging.info("Simulating route skipper decision in OceanWithBuoys environment...")

    # initialize storage of environment components and rewards
    info_list = []
    episode_buoys = deque([env.current_env[env.buoy_col]])
    episode_lats = deque([env.current_env[env.lat_col]])
    episode_lons = deque([env.current_env[env.lon_col]])
    episode_dates = deque([env.current_env[env.date_col]])
    episode_times = deque([env.current_env[env.time_col]])
    episode_rewards = deque([0])

    truncated, terminated = False, False

    # generate the episode following provided actions (buoys) in environment
    available_buoys = np.unique(buoykey_array)
    # exclude first buoy from those available to select
    available_buoys = available_buoys[available_buoys != start_env[env.buoy_col]]
    while not truncated and not terminated:
        # score all available buoys
        score = pd.Series(index=available_buoys, dtype=float)
        for buoy in available_buoys:
            # estimate travel time
            _, geodesic_days, _, _ = great_circle_travel_time(
                lat_start=env.current_env[env.lat_col],
                lon_start=env.current_env[env.lon_col],
                lat_end=env.current_env[f"LATITUD_{buoy}"],
                lon_end=env.current_env[f"LONGITUD_{buoy}"],
                velocity=env.inner_conf["geodesic_path"]["velocity"],
            )
            travel_hours = 24 * geodesic_days

            # get stop time
            stop_hours = env.conf["reward"]["stop_hours"]

            # get capture (+- synthetic_catch_delta) attending environment restrictions
            buoy_tons = env.current_env[f"PREDICTION_{buoy}"]
            # limit single capture
            if (
                env.conf["reward"]["capture_clip"]
                and buoy_tons > env.conf["reward"]["capture_clip"]
            ):
                buoy_tons = env.conf["reward"]["capture_clip"]
            # limit the capture by specified vessel max_tons
            if (env.caught_tons + buoy_tons) > env.conf["max_tons"]:
                buoy_tons = env.conf["max_tons"] - env.caught_tons

            # compute the score
            score[buoy] = buoy_tons - env.conf["reward"]["time_weight"] * (
                travel_hours + stop_hours
            )

        # head towards the buoy with maximum score
        # and mask it from selecting it again by avoiding its scoring in next step
        nextbuoy = score.idxmax()

        logging.debug(
            f"Next buoy predicted catch: {env.current_env[f'PREDICTION_{nextbuoy}']}"
        )

        # go to terminal buoy if there is no score greater than 0
        if (
            terminal_if_negative_score and score.max() < 0
        ) or env.caught_tons >= env.conf["max_tons"]:
            nextbuoy = terminal_buoy
        else:
            available_buoys = available_buoys[available_buoys != nextbuoy]

        # take the action in environment (through its action number of the environment)
        _, reward, terminated, truncated, info = env.step(
            env.action_name_to_idx(nextbuoy)
        )

        logging.info(
            f"Selected buoy: {nextbuoy}\n"
            f"Predicted score: {score.get(nextbuoy, np.nan)}\n"
            f"Actual reward: {reward}"
        )

        # saved retrieved environment components and rewards
        episode_buoys.append(env.current_env[env.buoy_col])
        episode_lats.append(env.current_env[env.lat_col])
        episode_lons.append(env.current_env[env.lon_col])
        episode_dates.append(env.current_env[env.date_col])
        episode_times.append(env.current_env[env.time_col])
        episode_rewards.append(reward)

        # store output info
        info_list.append(info)

        logging.debug(f"""Actual reward info: Capture: {info['capture_tons']},
            Journey hours: {info['journey_hours']}, Stop hours: {info['stop_hours']}""")

    # --------------------------------------
    # output relevant info
    logging.info(f"Episode return: {sum(episode_rewards)}")

    # retrieve info of the route
    capture_tons, journey_hours, stop_hours = [], [], []
    for info in info_list:
        capture_tons.append(info["capture_tons"])
        journey_hours.append(info["journey_hours"])
        stop_hours.append(info["stop_hours"])

    route_info = {
        "all_buoys": env.action_space.n,
        "n_buoys": len(episode_buoys),
        "capture": sum(capture_tons),
        "t_travel": sum(np.array(journey_hours) / 24),
        "t_stop": sum(np.array(stop_hours) / 24),
    }

    # store dataframe of visited buoys
    df_visited_buoys = pd.DataFrame(
        {
            LAT_COL: episode_lats,
            LON_COL: episode_lons,
            REW_COL: episode_rewards,
            KEY_COL: episode_buoys,
            DATE_COL: episode_dates,
            TIME_COL: episode_times,
        }
    )

    # filter records of environment buoys
    buoys_data_filtered = buoys_data[buoys_data[KEY_COL].isin(buoykey_array)]
    # get first record in time of environment buoys
    df_env_buoys = buoys_data_filtered.sort_values(
        by=[f"{DATE_COL}_{TIME_COL}"]
    ).drop_duplicates(subset=[KEY_COL], keep="first")
    # in addition, exclude visited buoys from environment buoys to plot
    df_env_buoys_not_visited = df_env_buoys[
        ~df_env_buoys[KEY_COL].isin(df_visited_buoys[KEY_COL].unique())
    ]

    # --------------------------------------
    cfg_plot = load_conf("./RLroutes_PAPER/inner_config.toml")["plot"]
    mpl.rcParams["mathtext.fontset"] = cfg_plot["mathtext_fontset"]
    mpl.rcParams["font.family"] = cfg_plot["font_family"]

    mpl.rcParams["xtick.labelsize"] = cfg_plot["xtick_labelsize"]
    mpl.rcParams["ytick.labelsize"] = cfg_plot["ytick_labelsize"]
    mpl.rcParams["axes.labelsize"] = cfg_plot["axes_labelsize"]

    # initialize plot objects and arrange plot data
    route_plot_1 = RL_routes_plot(
        lat_col=LAT_COL,
        lon_col=LON_COL,
        reward_col=REW_COL,
        key_col=KEY_COL,
        date_col=DATE_COL,
        time_col=TIME_COL,
        save_path=Path(f"{path_out}/route.png"),
    )
    route_plot_2 = copy.deepcopy(route_plot_1)

    # tune figsizes
    route_plot_1.fig.set_size_inches(8, 5)
    route_plot_2.fig.set_size_inches(8, 5)

    # plot with text with the visit order and buoy key
    route_plot_1.plot_buoys(
        df_env_buoys_not_visited,
        df_visited_buoys,
        text=Buoys_plot_text.VISIT_nd_KEY,
        title="Skipper",
        save=True,
    )

    # with datetime at which each point was visited
    route_plot_2.plot_buoys(
        df_env_buoys_not_visited,
        df_visited_buoys,
        text=Buoys_plot_text.DATETIME,
        more_info=route_info,
        title="Skipper",
        save=True,
    )

    # close employed environment
    env.close()

    print("SKIPPER TRAVEL IS DONE!")


def maskedPPO_OceanWithBuoys(
    train: bool,
    episode_eval: bool,
    start_loc: pd.Series,
    termination: Termination = Termination.LATLON,
    reward_end_episode: bool = False,
    use_masking: bool = True,
    repeat_buoys: Repeat_buoys = Repeat_buoys.ZERO_TONS,
    monitor_train: bool = True,
    env_global_obs: bool = False,
    env_n_states_stack: int | None = None,
    verbose: bool = False,
    model_path: Path = Path("./models/maskedppo_oceanwithbuoys"),
    conf_path: str = "./RLroutes_PAPER/config.toml",
):
    """Train and predict with Masked PPO agent in OceanWithBuoys env.

    Warnings
    --------
    In order to not modify Stable-Baselines3, info output of training steps will
    not be retrieved.

    See Also
    --------
    maskablePPO_train : Train an agent with StableBaselines3 Contrib Maskable PPO.
    maskablePPO_episode : Simulate an episode with a trained agent.
    """
    # create output folder if it does not exist
    Path(model_path).mkdir(parents=True, exist_ok=True)

    ppo_cfg = load_conf(conf_path)["agent"]["PPO"]

    # ------ INIT DATA ------
    buoykey_array, start_env, buoys_data, df_terminal_latlon = OceanWithBuoys_init(
        start_loc=start_loc,
        termination=termination,
        env_global_obs=env_global_obs,
        seed_synthetic_catch=ppo_cfg["seed"],
        buoy_col=KEY_COL,
        lat_col=LAT_COL,
        lon_col=LON_COL,
        date_col=DATE_COL,
        time_col=TIME_COL,
        datetime_col=DATETIME_COL,
        capture_col=CAPTURE_COL,
        conf_path=conf_path,
    )
    buoys_data_filtered = buoys_data[
        buoys_data[KEY_COL].isin(buoykey_array)
    ]  # filter records of environment buoys for later plots

    # ------ TRAIN ------
    if train:
        logging.info("Training...")
        _, t_train, n_expl_routes, t_optuna = maskablePPO_train(
            env=OceanWithBuoys,
            ppo_cfg=ppo_cfg,
            verbose=verbose,
            use_masking=use_masking,
            monitor_train=monitor_train,
            path_out=model_path,
            buoykey_array=buoykey_array,  # OceanWithBuoys inputs
            buoys_data=buoys_data,
            global_obs=env_global_obs,
            n_states_stack=env_n_states_stack,
            start_env=start_env,
            repeat_buoys=repeat_buoys,
            termination=termination,
            reward_end_episode=reward_end_episode,
            df_terminal_latlon=df_terminal_latlon,
            conf_path=conf_path,
        )

    # ------ EPISODE EVALUATION ------
    if episode_eval:
        logging.info("Simulating an episode with trained agent...")
        env = OceanWithBuoys(
            buoykey_array=buoykey_array,
            buoys_data=buoys_data,
            global_obs=env_global_obs,
            n_states_stack=env_n_states_stack,
            start_env=start_env,
            repeat_buoys=repeat_buoys,
            termination=termination,
            reward_end_episode=reward_end_episode,
            df_terminal_latlon=df_terminal_latlon,
            conf_path=conf_path,
        )
        # notice reset seed not specified,
        # so a seed will be chosen from some source of entropy
        env.reset()
        agent = MaskablePPO.load(f"{model_path}/{model_path.stem}", env=env)

        (
            episode_buoys,
            episode_lats,
            episode_lons,
            episode_dates,
            episode_times,
            episode_rewards,
            info_list,
        ) = maskablePPO_episode(
            start_state=(
                start_env
                if env.n_states_stack is None
                else np.array(env._norm_states_memory.memory).flatten()
            ),  # CAUTION: if modified, make sure states stack has dtype `State_norm`
            agent=agent,
            env=env,
            deterministic=True,
            use_masking=use_masking,
        )

        # save final policy episode
        RL_plot_1 = RL_routes_plot(
            lat_col=LAT_COL,
            lon_col=LON_COL,
            reward_col=REW_COL,
            key_col=KEY_COL,
            date_col=DATE_COL,
            time_col=TIME_COL,
            save_path=Path(f"{model_path}/episode"),
        )
        RL_plot_2 = copy.deepcopy(RL_plot_1)

        df_env_buoys = buoys_data_filtered.sort_values(
            by=[f"{DATE_COL}_{TIME_COL}"]
        ).drop_duplicates(
            subset=[KEY_COL], keep="first"
        )  # get first record in time of environment buoys
        df_visited_buoys = pd.DataFrame(
            {
                LAT_COL: episode_lats,
                LON_COL: episode_lons,
                REW_COL: episode_rewards,
                KEY_COL: episode_buoys,
                DATE_COL: episode_dates,
                TIME_COL: episode_times,
            }
        )

        # retrieve reward info from env steps info
        capture_tons, journey_hours, stop_hours = [], [], []
        for info in info_list:
            capture_tons.append(info["capture_tons"])
            journey_hours.append(info["journey_hours"])
            stop_hours.append(info["stop_hours"])
        # add extra info to the final policy episode
        extra_info = {
            "all_buoys": env.action_space.n,
            "n_buoys": len(episode_buoys),
            "capture": sum(capture_tons),
            "t_travel": sum(np.array(journey_hours) / 24),
            "t_stop": sum(np.array(stop_hours) / 24),
        }
        if train:
            extra_info["t_train"] = t_train
            extra_info["n_episodes"] = n_expl_routes
            if t_optuna is not None:
                extra_info["t_optimization"] = t_optuna

        # plot with text with the visit order and buoy key
        RL_plot_1.plot_buoys(
            df_env_buoys,
            df_visited_buoys,
            text=Buoys_plot_text.VISIT_nd_KEY,
            more_info=extra_info,
            save=True,
        )
        # plot with datetime at which each point was visited
        RL_plot_2.plot_buoys(
            df_env_buoys,
            df_visited_buoys,
            text=Buoys_plot_text.DATETIME,
            save=True,
        )
        env.close()

    print("MASKED PPO IS DONE!")


if __name__ == "__main__":
    # ------ INIT PARAMETERS ------
    repeat_buoys = Repeat_buoys.PROHIBITED
    termination = Termination.LATLON
    reward_end_episode = False
    env_global_obs = False
    env_n_states_stack = None

    # ------ INIT DATA ------
    start_datetime = dt.datetime(2023, 1, 1, 10, 10, 10)
    start_loc = pd.Series(
        {
            KEY_COL: "b 0",
            LAT_COL: 2.6,
            LON_COL: -149.85,
            DATE_COL: start_datetime.date(),
            TIME_COL: start_datetime.time(),
        }
    )

    # ------ SIMULATIONS ------
    maskedPPO_OceanWithBuoys(
        train=False,
        episode_eval=True,
        start_loc=start_loc,
        termination=termination,
        reward_end_episode=reward_end_episode,
        repeat_buoys=repeat_buoys,
        env_global_obs=env_global_obs,
        env_n_states_stack=env_n_states_stack,
    )
    random_OceanWithBuoys(
        train=False,
        repeat_buoys=repeat_buoys,
        start_loc=start_loc,
        termination=termination,
        run_seeds=[None],
    )
    skipper_OceanWithBuoys(
        start_loc=start_loc,
        terminal_buoy="b 28",
        terminal_if_negative_score=False,
    )
