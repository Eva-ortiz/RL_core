import logging
import random
import sys
from collections import deque
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from agent_predict import maskablePPO_episode
from agent_train import maskablePPO_train
from global_vars import WORKING_DIR
from RL.DRL import DQN_agent, DRL_agent  # noqa E402
from sb3_contrib import MaskablePPO
from utils import load_conf, set_logging, timer

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
    not be retrieved.

    See Also
    --------
    maskablePPO_train : Train an agent with StableBaselines3 Contrib Maskable PPO.
    maskablePPO_episode : Simulate an episode with a trained agent.
    """
    # ------ INIT DATA ------
    start_env = "Developer insert value: Environment"

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

        # Developer instruction: post-process relevant data:
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
    train: bool,
    greedy_eval: bool,
    algorithm: Literal["Sarsa", "Q-learning", "double_Q-learning"],
    monitor_train: bool = True,
    logging_level: str = "warn",
    model_path: Path | None = None,
    conf_path: str = "./config.toml",
    seed: int | None = None,
    **env_kwargs,
):
    """RL approximated method simulation of the agent in environment.

    We employ neural networks to perform this approximation.

    Parameters
    ----------
    env : gym.Env
        Environment class to run the input agent.
    train: bool
        Train the agent or load it from `model_path`.
    greedy_eval: bool
        Perform a greedy episode for agent evaluation and save its info.
    algorithm : Literal["Sarsa", "Q-learning", "double_Q-learning"]
        Determine the RL algorithm to use. Integrated algorithms are:
            * Sarsa
            * Q-learning
            * double Q-learning
    monitor_train : bool, optional
        Monitor training and save monitorization data and plots.
        By default, True.
    logging_level : str, optional
        Select logging level. Additionally, if `debug`/`info` are selected, set
        verbose to True, else, to False. By default, `warn`.
    model_path : Path | None, optional
        Parent path where agents and their evaluation info are saved. If None,
        default folder of each algorithm. By default, None.
    conf_path : str, optional
        Configuration file path. By default, Path("./config.toml").
    seed : int | None, optional
        Seed number along the simulation. If None, do not fix any seed.
        By default, None.
    **env_kwargs
        Inputs forwarded to the environment constructor (``env(**env_kwargs)``),
        such as ``start_env``, ``global_obs`` or ``n_states_stack``.

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

    References
    ----------
    .. [1] https://stackoverflow.com/questions/38537905/set-logging-levels
    .. [2] https://docs.python.org/3/library/logging.html#levels
    """
    # set logging level [1], [2]
    set_logging(level=logging_level)
    verbose = logging_level in ["debug", "info"]

    # load the configuration
    cfg = load_conf(conf_path)
    cfg_logging = cfg["logging"]
    cfg_env = cfg["environment"]
    cfg_hiperpar = cfg["DRL"][algorithm]["hiperparams"]

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

    # select RL algorithm
    agent_class = DQN_agent if algorithm == "double_Q-learning" else DRL_agent

    if train:
        # create the RHT environment
        train_env = env(**env_kwargs)

        # create the new agent with selected algorithm
        agent = agent_class(
            algorithm=algorithm,
            actions=list(train_env._action_name_dict.values()),
            save_folder=model_path,
            verbose=verbose,
            hidden_layers=cfg_hiperpar["hidden_layers"],
            hidden_neur=cfg_hiperpar["hidden_neur"],
        )

        with timer(tag="train_time") as train_time:
            agent.train(
                device=device,
                environment=train_env,
                plot_learning_curves=monitor_train,
                save_q_net=True,
                reward_curve_mode=cfg_env["plots"]["reward_curve_mode"],
                reward_curve_steps_per_point=cfg_env["plots"][
                    "reward_curve_steps_per_point"
                ],
                debug_counter=cfg_logging["debug_counter"],
                seed=seed,
                **cfg_hiperpar["train"],
            )

        logging.info(f"Training time: {train_time():.2f} s")
        logging.info(f"""Number of visited states: {len(agent.visited_states_norm)}""")

    # ------------- output relevant data -------------
    if greedy_eval:
        eval_env = env(**env_kwargs)

        # create the new agent with selected algorithm
        agent = agent_class(
            algorithm=algorithm,
            actions=list(eval_env._action_name_dict.values()),
            save_folder=f"{algorithm}_results",
            verbose=verbose,
            hidden_layers=cfg_hiperpar["hidden_layers"],
            hidden_neur=cfg_hiperpar["hidden_neur"],
        )
        # load its network
        agent.load_net(
            in_dim=eval_env.observation_space.shape[0],
            device=device,
            model_path=model_path,
        )

        agent.greedy_simulation(
            q_net=agent.q_net,
            env=eval_env,  # [3]
            max_steps=cfg_env["max_transitions"],
            device=device,
            reset_options=cfg_hiperpar["train"].get("reset_options", None),
        )

    print("DEEP REINFORCEMENT LEARNING IS DONE!")


def random_search(
    timesteps: int, env: gym.Env, seed: int | None = None
) -> tuple[list, list, list, list, list, list, list, list]:
    """Discover the best episode from random search in input env.

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
    best_state1_sequence : list
        Best episode feature 1 of the state, in order.
    best_state2_sequence : list
        Best episode feature 2 of the state, in order.
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
    best_state1_sequence, best_state2_sequence, best_info_sequence = [], [], []
    return_sequence, episode_step_sequence, n_exp_sequence = [], [], []

    # sequence of values to keep track of performed actions
    # start the track from initialized environment
    action_seq = deque([env.current_env[env.action_col]])
    reward_seq = deque([0])
    state1_seq = deque([env.current_env[env.state_col_1]])  # [4 EXAMPLE]
    state2_seq = deque([env.current_env[env.state_col_2]])  # [4 EXAMPLE]
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
        state1_seq.append(env.current_env[env.state_col_1])  # [4 EXAMPLE]
        state2_seq.append(env.current_env[env.state_col_2])  # [4 EXAMPLE]
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
                best_state1_sequence = list(state1_seq)  # [4 EXAMPLE]
                best_state2_sequence = list(state2_seq)  # [4 EXAMPLE]
                best_info_sequence = info_list
                logging.info(
                    f"Best episode with return {rl_return} obtained in step {step}."
                )

            # save return track
            return_sequence.append(rl_return)
            episode_step_sequence.append(step)
            n_exp_sequence.append(n_exp_episode)

            # reset tracked episode
            action_seq = deque([env.current_env[env.action_col]])
            reward_seq = deque([0])
            state1_seq = deque([env.current_env[env.state_col_1]])  # [4 EXAMPLE]
            state2_seq = deque([env.current_env[env.state_col_2]])  # [4 EXAMPLE]
            info_list = []
            n_exp_episode = 0

    return (
        best_action_sequence,
        best_reward_sequence,
        best_state1_sequence,  # [4 EXAMPLE]
        best_state2_sequence,  # [4 EXAMPLE]
        best_info_sequence,
        return_sequence,
        episode_step_sequence,
        n_exp_sequence,
    )


def random_agent(
    env: gym.Env,
    train: bool,
    run_seeds: int | list[int | None] | None = None,
    model_path: Path = Path("./models/random"),
    conf_path: str = "./config.toml",
    **env_kwargs,
):
    """Discover the best episode from random exploration within `env`.

    See Also
    --------
    random_search : algorithm to perform the random search in the environment.
    """
    if run_seeds is None:
        run_seeds = [None]
    elif isinstance(run_seeds, int):
        run_seeds = [run_seeds]

    if train:
        cfg_random = load_conf(conf_path)["agent"]["random"]
        search_env = env(**env_kwargs)

        # ------ EPISODE SEARCH ------
        logging.info("Looking for the best episode with random search...")
        with timer(tag="train_time") as train_time:
            results = []
            for seed in run_seeds:
                logging.info(f"\n\tSeed {seed}\n")
                (
                    best_episode_actions,
                    best_episode_rewards,
                    best_episode_state1s,
                    best_episode_state2s,
                    best_episode_info,
                    return_sequence,
                    step_sequence,
                    n_exp_sequence,
                ) = random_search(
                    timesteps=cfg_random["timesteps"], env=search_env, seed=seed
                )
                results.append(
                    {
                        "best_actions": best_episode_actions,
                        "best_rewards": best_episode_rewards,
                        "best_state1s": best_episode_state1s,  # [4 EXAMPLE]
                        "best_state2s": best_episode_state2s,  # [4 EXAMPLE]
                        "returns": return_sequence,
                        "experiences": n_exp_sequence,
                        "steps": step_sequence,
                        "best_infos": best_episode_info,
                    }
                )
                # Developer instruction: post-process relevant info of best episode
            df_results = pd.DataFrame(results)
            df_results.to_csv(f"{model_path}/random_train_results_seed_{run_seeds}.csv")
        logging.info(f"Search time: {train_time():.2f} s")
        search_env.close()

    print("RANDOM SEARCH IS DONE!")


if __name__ == "__main__":
    maskedPPO_agent(
        env="Developer insert value: gym.Env class (not object)",
        train=False,  # allow to train
        episode_eval=True,  # allow for greedy episode
        use_masking="Developer instruction: set True/False depending on the env",
        monitor_train=True,
        verbose=False,
        env_kwargs="Developer insert value: env kwargs dict",
    )
    approximated_simulation(
        env="Developer insert value: gym.Env class",
        train=False,  # allow to train
        greedy_eval=True,  # allow for greedy episode
        algorithm=(
            "Developer insert value: "
            "Literal['Sarsa', 'Q-learning', 'double_Q-learning']"
        ),
        monitor_train=True,
        logging_level="warn",
        seed=None,
        env_kwargs="Developer insert value: env kwargs dict",
    )
    random_agent(
        env="Developer insert value: gym.Env class (not object)",
        train=True,  # allow to perform the random search
        run_seeds=None,
        env_kwargs="Developer insert value: env kwargs dict",
    )
