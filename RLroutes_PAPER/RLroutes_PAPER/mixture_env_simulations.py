import copy
import datetime as dt
import logging
from pathlib import Path

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
from RLroutes_PAPER.mixture_env import MixtureEnv3
from RLroutes_PAPER.utils.conf import load_conf
from RLroutes_PAPER.utils.plots import Buoys_plot_text, RL_routes_plot

LAT_COL = "LATITUD"
LON_COL = "LONGITUD"
REW_COL = "REWARD"
KEY_COL = "KEY_BALIZA"
DATE_COL = "FECHA"
TIME_COL = "HORA"
DATETIME_COL = "FECHA_HORA"
CAPTURE_COL = "CAPTURA"


def maskedPPO_OceanWithBuoys_mixture(
    train: bool,
    episode_eval: bool,
    env1_start_loc: pd.Series,
    env2_start_loc: pd.Series,
    env3_start_loc: pd.Series,
    termination: Termination = Termination.LATLON,
    reward_end_episode: bool = False,
    use_masking: bool = True,
    repeat_buoys: Repeat_buoys = Repeat_buoys.ZERO_TONS,
    monitor_train: bool = True,
    env_global_obs: bool = False,
    env_n_states_stack: int | None = None,
    verbose: bool = False,
    model_path: Path = Path("./models/maskedppo_mixture_oceanwithbuoys"),
    conf_path: str = "./RLroutes_PAPER/config.toml",
    env1_conf_path: str = "./RLroutes_PAPER/config_1.toml",
    env2_conf_path: str = "./RLroutes_PAPER/config_2.toml",
    env3_conf_path: str = "./RLroutes_PAPER/config_3.toml",
):
    """Train and predict with Masked PPO agent in a mixture of OceanWithBuoys envs.

    Warnings
    --------
    * In order to not modify Stable-Baselines3, info output of training steps
      will not be retrieved.
    * Optuna hyperparameter optimization is not compatible with the mixture
      environment. Provide fixed (non list) hyperparameters in `conf_path`.
    * Training requires vectorized environments with multiprocessing
      (`n_envs` > 0 and `env_multiprocess = true` in `conf_path`). Every
      `MixtureEnv3` built during training wraps the SAME `env1`, `env2` and
      `env3` objects, which store the episode state (current position,
      caught tons, visited buoys...). With `SubprocVecEnv` each worker receives
      its own pickled copy of them, but with `DummyVecEnv` all workers share
      them in the same process, and with `n_envs = 0` the training env shares
      them with the greedy check env. In both cases, a reset or step of one env
      silently modifies the ongoing episode of another one.

    See Also
    --------
    maskablePPO_train : Train an agent with StableBaselines3 Contrib Maskable PPO.
    maskablePPO_episode : Simulate an episode with a trained agent.
    """
    # create output folder if it does not exist
    Path(model_path).mkdir(parents=True, exist_ok=True)

    ppo_cfg = load_conf(conf_path)["agent"]["PPO"]

    # ------ INIT DATA ------
    OceanWithBuoys_init_kwargs = {
        "termination": termination,
        "env_global_obs": env_global_obs,
    }

    OceanWithBuoys_kwargs = {
        "termination": termination,
        "n_states_stack": env_n_states_stack,
        "repeat_buoys": repeat_buoys,
        "global_obs": env_global_obs,
        "reward_end_episode": reward_end_episode,
    }

    # ENV1 -----------
    env1_buoykey_array, env1_start_env, env1_buoys_data, env1_df_terminal_latlon = (
        OceanWithBuoys_init(
            start_loc=env1_start_loc,
            seed_synthetic_catch=ppo_cfg["seed"],
            conf_path=env1_conf_path,
            **OceanWithBuoys_init_kwargs,
        )
    )
    env1_buoys_data_filtered = env1_buoys_data[
        env1_buoys_data[KEY_COL].isin(env1_buoykey_array)
    ]  # filter records of environment buoys for later plots

    env1 = OceanWithBuoys(
        buoykey_array=env1_buoykey_array,
        buoys_data=env1_buoys_data,
        start_env=env1_start_env,
        df_terminal_latlon=env1_df_terminal_latlon,
        conf_path=env1_conf_path,
        **OceanWithBuoys_kwargs,
    )

    # ENV2 -----------
    env2_buoykey_array, env2_start_env, env2_buoys_data, env2_df_terminal_latlon = (
        OceanWithBuoys_init(
            start_loc=env2_start_loc,
            seed_synthetic_catch=ppo_cfg["seed"] + 1,  # one seed higher than env1
            conf_path=env2_conf_path,
            **OceanWithBuoys_init_kwargs,
        )
    )
    env2_buoys_data_filtered = env2_buoys_data[
        env2_buoys_data[KEY_COL].isin(env2_buoykey_array)
    ]  # filter records of environment buoys for later plots

    env2 = OceanWithBuoys(
        buoykey_array=env2_buoykey_array,
        buoys_data=env2_buoys_data,
        start_env=env2_start_env,
        df_terminal_latlon=env2_df_terminal_latlon,
        conf_path=env2_conf_path,
        **OceanWithBuoys_kwargs,
    )

    # ENV3 -----------
    env3_buoykey_array, env3_start_env, env3_buoys_data, env3_df_terminal_latlon = (
        OceanWithBuoys_init(
            start_loc=env3_start_loc,
            seed_synthetic_catch=ppo_cfg["seed"] + 1,  # one seed higher than env1
            conf_path=env3_conf_path,
            **OceanWithBuoys_init_kwargs,
        )
    )
    env3_buoys_data_filtered = env3_buoys_data[
        env3_buoys_data[KEY_COL].isin(env3_buoykey_array)
    ]  # filter records of environment buoys for later plots

    env3 = OceanWithBuoys(
        buoykey_array=env3_buoykey_array,
        buoys_data=env3_buoys_data,
        start_env=env3_start_env,
        df_terminal_latlon=env3_df_terminal_latlon,
        conf_path=env3_conf_path,
        **OceanWithBuoys_kwargs,
    )

    # ------ TRAIN ------
    if train:
        # sub-environments must not be shared among training envs (see Warnings)
        if ppo_cfg["n_envs"] == 0 or not ppo_cfg["env_multiprocess"]:
            raise ValueError(
                "Mixture training requires `n_envs` > 0 and "
                "`env_multiprocess = true`, otherwise sub-environments are "
                "shared among environments."
            )
        logging.info("Training...")

        mixprob = 1 / 3
        _, t_train, n_expl_routes, t_optuna = maskablePPO_train(
            env=MixtureEnv3,
            ppo_cfg=ppo_cfg,
            verbose=verbose,
            use_masking=use_masking,
            monitor_train=monitor_train,
            path_out=model_path,
            **{"env1": env1, "env2": env2, "env3": env3, "mix_prob": mixprob},
        )

    # ------ EPISODE EVALUATION ------
    if episode_eval:
        eval_objects = {
            "env1": {
                "start_env": env1_start_env,
                "buoys_data_filtered": env1_buoys_data_filtered,
            },
            "env2": {
                "start_env": env2_start_env,
                "buoys_data_filtered": env2_buoys_data_filtered,
            },
            "env3": {
                "start_env": env3_start_env,
                "buoys_data_filtered": env3_buoys_data_filtered,
            },
        }

        for idx, env in enumerate([env1, env2, env3]):
            env_name = f"env{idx+1}"
            logging.info(
                f"Simulating an episode in {env_name} with trained mixture agent..."
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
                    eval_objects[env_name]["start_env"]
                    if env.n_states_stack is None
                    else np.array(env._norm_states_memory.memory).flatten()
                ),  # CAUTION: if modified, make sure states stack dtype is `State_norm`
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
                save_path=Path(f"{model_path}/{env_name}_episode"),
            )
            RL_plot_2 = copy.deepcopy(RL_plot_1)

            df_env_buoys = (
                eval_objects[env_name]["buoys_data_filtered"]
                .sort_values(by=[f"{DATE_COL}_{TIME_COL}"])
                .drop_duplicates(subset=[KEY_COL], keep="first")
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
    env_global_obs = True
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
    maskedPPO_OceanWithBuoys_mixture(
        train=True,
        episode_eval=True,
        env1_start_loc=start_loc,
        env2_start_loc=start_loc,
        env3_start_loc=start_loc,
        termination=termination,
        reward_end_episode=reward_end_episode,
        repeat_buoys=repeat_buoys,
        env_global_obs=env_global_obs,
        env_n_states_stack=env_n_states_stack,
    )
