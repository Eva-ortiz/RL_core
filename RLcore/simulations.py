import copy
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from agent_predict import maskablePPO_episode
from agent_train import maskablePPO_train
from global_vars import WORKING_DIR
from sb3_contrib import MaskablePPO

from RLcore.RL.DRL import DQN_agent, DRL_agent  # noqa E402
from RLcore.RL.utils import exists, load_conf, set_logging, store_series_to_hdf, timer

# for HPC, specify the path to import from our modules
# .. [1] https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(1, WORKING_DIR)  # [1]


def maskedPPO_agent(
    env: gym.Env,
    train: bool,
    episode_eval: bool,
    use_masking: bool = True,
    monitor_train: bool = True,
    verbose: bool = False,
    model_path: Path = Path("./models/maskedppo"),
    conf_path: str = "./config.toml",
    **env_kwargs,
):
    """Train and predict with Masked PPO agent within `env` environment.

    Warnings
    --------
    In order to not modify Stable-Baselines3, info output of training steps will
    not be retrieved. However, during training:
    * Missing uv data can be tracked setting logging level to DEBUG.

    See Also
    --------
    maskablePPO_train : Train an agent with StableBaselines3 Contrib Maskable PPO.
    maskablePPO_episode : Simulate an episode with a trained agent.
    """
    # ------ INIT DATA ------
    start_env = "TODO : Environment"

    # ------ TRAIN ------
    if train:
        logging.info("Training...")
        _, t_train, n_expl_episodes, t_optuna = maskablePPO_train(
            env=env,
            verbose=verbose,
            use_masking=use_masking,
            monitor_train=monitor_train,
            cfg_path=conf_path,
            path_out=model_path,
            **env_kwargs,
        )

    # ------ EPISODE EVALUATION ------
    if episode_eval:
        logging.info("Simulating a greedy episode with trained agent...")
        env = env(**env_kwargs)
        # notice reset seed not specified,
        # so a seed will be chosen from some source of entropy
        env.reset()
        agent = MaskablePPO.load(f"{model_path}/{model_path.stem}", env=env)

        (
            episode_actions,
            episode_state1s,
            episode_state2s,
            episode_rewards,
            info_list,
        ) = maskablePPO_episode(
            start_state=start_env,
            agent=agent,
            env=env,
            deterministic=True,
            use_masking=use_masking,
        )

        # TODO: post-process relevant data:
        #   - policy representation: episode_actions, episode_state1s, episode_state2s
        #   - policy data : episode_rewards and return, training times, number
        #     of explored episodes
        #   - info_list

        # retrieve specific info from env steps info
        info1, info2, info3 = [], [], []
        for info in info_list:
            info1.append(info["info1"])  # [4 EXAMPLE]
            info2.append(info["info2"])  # [4 EXAMPLE]
            info3.append(info["info3"])  # [4 EXAMPLE]

        extra_info = {
            "all_actions": env.action_space.n,
            "episode_acions": len(episode_actions),
            "return": sum(episode_rewards),
        }
        if train:
            extra_info["t_train"] = t_train
            extra_info["n_episodes"] = n_expl_episodes
            if t_optuna is not None:
                extra_info["t_optimization"] = t_optuna

    print("MASKED PPO IS DONE!")


def approximated_simulation(
    env: gym.Env,
    algorithm: Literal["Sarsa", "Q-learning", "double_Q-learning"],
    cfg_path: Path = Path("./RLcore/config.toml"),
    save_computed_htc: bool = False,
    save_htc_csv: bool = True,
    save_q_net: bool = True,
    plot: bool = False,
    logging_level: str = "warn",
    seed: int | None = None,
):
    """RL approximated method simulation of the agent in environment.

    We employ neural networks to perform this approximation.

    Parameters
    ----------
    env : gym.Env
        Environment class to run the input agent.
    algorithm : Literal["Sarsa", "Q-learning", "double_Q-learning"]
        Determine the RL algorithm to use. Integrated algorithms are:
            * Sarsa
            * Q-learning
            * double Q-learning
    cfg_path : Path, optional
        Configuration file path. By default, Path("./config.toml").
    save_computed_htc : bool, optional
        Overwrite stored htc values in order to add new computed values. If
        several runs are executed at the same time, this could lead to execution
        failure.
        By default, False.
    save_htc_csv : bool, optional
        Select to store a csv with the htc value of each state. Useful to
        visualize those values. By default, True.
    save_q_net : bool, optional
        Select to save the action-state values network.
        By default, True.
    plot : bool, optional
        Select if we desire to obtain different plots, such as:
         - Loss curves
         - Reward curves
        By default, False.
    logging_level : str, optional
        Select logging level. Additionally, if `debug`/`info` are selected, set
        verbose to True, else, to False.
        By default, `warn`.
    seed : int | None, optional
        Seed number along the simulation. If None, do not fix any seed.
        By default, None.

    Returns
    -------
    agent : agent
        Trained agent.
    env : environment
        Environment where the agent was trained.
    last_state : str
        Predicted optimal state.
        We will assume optimal state is the state where the agent tends to end.
        Because of that, we will perform a very long simulation with a greedy
        policy and output the end state as the OPTIMAL STATE.

    Warnings
    --------
    `HTC_{n_layers}_layers.h5` input can be problematic.
        If it has been previously **UNsuccessfully** generated, a HDF5ExtError
        will raise.
        To solve it, delete `HTC_{n_layers}_layers.h5` and execute twice
        the simulation, as the first modification to the file will raise a
        ValueError as file is already opened and it is required to be in
        read-only mode.
        Another solution is to delete the problematic file and copy-paste it
        from HTC_backup.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/38537905/set-logging-levels
    .. [2] https://stackoverflow.com/questions/65728037/matplotlib-debug-turn-off-when-python-debug-is-on-to-debug-rest-of-program
    .. [3] https://docs.python.org/3/library/copy.html
    .. [4] https://docs.python.org/3/library/logging.html#levels
    """
    # set logging level [1], [4]
    set_logging(level=logging_level)

    verbose = logging_level in ["debug", "info"]

    # load the configuration
    cfg_logging = load_conf(cfg_path, key="logging")

    cfg = load_conf(cfg_path, key="RHT")
    cfg_plots = cfg["plots"]

    cfg = load_conf(cfg_path, key="DRL")
    cfg_algorithm = cfg[algorithm]
    cfg_hiperpar = cfg_algorithm["RHT"]["hiperparams"]
    cfg_save_folder = cfg_algorithm["RHT"]["generic_save_folder"]

    # We want to be able to train our model on a hardware accelerator like the
    # GPU or MPS, if available.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f"Using {device} device")

    # store start time of the program
    start_time = time.time()

    # load HTC input constants
    # frequencies range
    with open("./data/HTC/HTC_frequencies.npy", "rb") as f:
        freq_range = np.load(f)
    # materials permitivities at each frequencies
    with open("./data/HTC/HTC_permitivities.npy", "rb") as f:
        mat_per = np.load(f)

    # create the RHT environment
    env = env(mat_per, freq_range, cfg_path=cfg_path, seed=seed)

    # create the new agent with selected algorithm
    agent_class = DQN_agent if algorithm == "double_Q-learning" else DRL_agent

    agent = agent_class(
        algorithm=algorithm,
        actions=env.action_space,
        save_folder=cfg_save_folder,
        verbose=verbose,
        hidden_layers=cfg_hiperpar["hidden_layers"],
        hidden_neur=cfg_hiperpar["hidden_neur"],
    )

    agent.train(
        device=device,
        environment=env,
        discount_rate=cfg_hiperpar["discount_rate"],
        lr=cfg_hiperpar["lr"],
        reduce_perc_lr=cfg_hiperpar["reduce_perc_lr"],
        min_lr=cfg_algorithm["min_lr"],
        epsilon=cfg_hiperpar["epsilon"],
        reduce_eps=cfg_hiperpar["reduce_eps"],
        min_eps=cfg_algorithm["min_eps"],
        batch_size=cfg_hiperpar["batch_size"],
        max_steps=cfg_hiperpar["max_steps"],
        tol_loss=cfg_hiperpar["tol_loss"],
        plot_learning_curves=plot,
        save_q_net=save_q_net,
        memory_size=cfg_hiperpar.get("memory_size", None),  # only DQN
        n_batch_per_step=cfg_hiperpar.get("n_batch_per_step", None),  # only DQN
        n_new_experiences_per_step=cfg_hiperpar.get(
            "n_new_experiences_per_step", None
        ),  # only DQN
        target_estimation_mode=cfg_hiperpar.get(
            "target_estimation_mode", None
        ),  # only DQN
        n_steps_for_target_net_update=cfg_hiperpar.get(
            "n_steps_for_target_net_update", None
        ),  # only DQN
        episode_start_state=cfg_hiperpar["episode_start_state"],
        decorrelated=cfg_hiperpar["decorrelated"],
        step_count=cfg_hiperpar["step_count"],
        reward_curve_mode=cfg_plots["reward_curve_mode"],
        reward_curve_steps_per_point=cfg_plots["reward_curve_steps_per_point"],
        debug_counter=cfg_logging["debug_counter"],
        seed=seed,
    )

    # print execution time of this problem
    timer(start_time, time.time())

    # ------------- output relevant data -------------
    _, last_reward, _, last_state, last_action = agent.greedy_simulation(
        q_net=agent.q_net,
        environment=copy.deepcopy(env),  # [3]
        start_state=cfg_hiperpar["episode_start_state"],
        steps=env.n_layers * 10,
        device=device,
    )
    print(
        f"Last state is: {last_state}",
        f"\nWith an associated reward of: {f'{last_reward:.3f}'}",
    )
    logging.info(f"Associated last action was {last_action}")
    logging.info(
        f"Number of visited states: {len(agent.visited_states)}/{len(env.state_space)}"
        if exists(env.state_space)
        else ""
    )
    logging.info(
        f"Number of reached states: {len(agent.max_htc_vs_known_state.iterations)}"
        f"/{len(env.state_space)}"
        if exists(env.state_space)
        else ""
    )

    # store the obtained states htc into hdf and csv (if selected)
    if save_computed_htc:
        store_series_to_hdf(
            f"{env.htc_path}/HTC_{env.n_layers}_layers.h5", env.htc_values
        )

    if save_htc_csv:
        env.htc_values.to_csv(
            f"./data/{cfg_save_folder}/state_{env.n_layers}_layers_htc.csv"
        )

    return agent, env, last_state


if __name__ == "__main__":
    maskedPPO_agent(
        env="TODO : gym.Env class (not object)",
        train=False,  # allow to train
        episode_eval=True,  # allow for greedy episode
        use_masking="TODO : depending of the env",
        monitor_train=True,
        verbose=False,
        env_kwargs="TODO : env kwargs dict",
    )
    approximated_simulation(
        env="TODO : gym.Env class",
        algorithm="TODO: Literal['Sarsa', 'Q-learning', 'double_Q-learning']",
        save_q_net=True,
        plot=True,
        logging_level="warn",
        seed=None,
    )
