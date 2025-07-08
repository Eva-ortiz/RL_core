import logging
import re
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from agent_predict import maskablePPO_episode
from environment import call_method_or_attr_of_envs
from sb3_contrib import MaskablePPO
from sb3_custom.common.env_util import make_vec_env_custom
from sb3_custom.common.monitor import Monitor_custom
from sb3_custom.ppo_mask.ppo_mask import MaskablePPO_custom
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from utils import Capturing, check_bool_or_float, check_bool_or_int, load_conf, timer


def agent_training_outputs(
    learn_outputs: Logger,
    agent: MaskablePPO,
    path_out: Path,
):
    """Plot learning curves from Stable-Baselines3 agents.

    Parameters
    ----------
    learn_outputs : Logger
        Stable-Baselines3 agent logger with training data. Training data
        frequency is known to be `n_updates`, i.e., the number of `n_epochs`
        policy optimizations have been completed.
    agent : MaskablePPO
        Trained agent. Only used to check some characteristics that set which
        learning curves can be retrieved.
    path_out : Path
        Folder where monitor plots and csv will be saved.

    Warnings
    --------
    Notice all variables are given every n_epochs. This fact is due to
    MaskablePPO records the value / mean value of every n_epochs network updates
    for them.

    See Also
    --------
    MaskablePPO.train()
        Method where can be confirmed x axis is the number of times the
        `n_epochs` policy optimization has been completed.

    Examples
    --------
    >>> input `learn_outputs`
    ['-------------------------------------------',
    '| rollout/                |               |',
    '|    ep_len_mean          | 5             |',
    '|    ep_rew_mean          | -105          |',
    '| time/                   |               |',
    '|    fps                  | 3             |',
    '|    iterations           | 2             |',
    '|    time_elapsed         | 3             |',
    '|    total_timesteps      | 10            |',
    '| train/                  |               |',
    '|    approx_kl            | 0.00020577907 |',
    '|    clip_fraction        | 0             |',
    '|    clip_range           | 0.2           |',
    '|    entropy_loss         | -2.18         |',
    '|    explained_variance   | -0.00879      |',
    '|    greedy_test          | -37.6         |',
    '|    learning_rate        | 0.001         |',
    '|    loss                 | 572           |',
    '|    n_experiences        | 5             |',
    '|    n_updates            | 3             |',
    '|    policy_gradient_loss | -0.00826      |',
    '|    value_loss           | 1.15e+03      |',
    '-------------------------------------------']
    """
    # use captured outputs to create a dict with info to create learning curves
    learning_vars = [
        "n_updates",  # x axis of plots
        "n_experiences",  # x axis of plots
        "loss",
        "entropy_loss",
        "policy_gradient_loss",
        "value_loss",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "clip_range",
        "learning_rate",
        "greedy_test",
    ] + (["clip_range_vf"] if agent.clip_range_vf is not None else [])
    learning_curves = dict()
    # look for var names and their values in a string of format
    # "| name | value |" where value can be +-int, +-float, +-float*e+-int
    pattern = re.compile(
        r"\|\s*(\S+)\s*\|\s*([-+]?\d+|[-+]?\d+\.\d+|[-+]?\d+[eE][-+]?\d+|[-+]?\d+\.\d+[eE][-+]?\d+)\s*\|"
    )
    # read each `learn_outputs` row, extract name and value and store them if
    # name is in `learning_vars`. `learn_outputs` example in docstring
    for record in learn_outputs:
        matches = pattern.findall(record)
        for match in matches:
            # extract variable-value pairs
            variable, value = match
            # store them in a dict if they are `learning_vars`
            if variable in learning_vars:
                if variable in learning_curves:
                    learning_curves[variable].append(float(value))
                else:
                    learning_curves[variable] = [float(value)]

    # check all learning curves have the same length
    curves_lens = [len(record) for record in learning_curves.values()]
    assert all(
        curve_len == curves_lens[0] for curve_len in curves_lens
    ), f"Learning curves expected to be the same length, but given {curves_lens} len."

    # plot and save recorded training data
    x_updates = learning_curves.pop("n_updates")
    x_experiences = learning_curves.pop("n_experiences")
    df_training_updates = pd.DataFrame({"n_updates": x_updates})
    df_training_experiences = pd.DataFrame({"n_experiences": x_experiences})

    # plot variables as a function of n_updates
    for var_name, values in learning_curves.items():
        if var_name == "greedy_test":
            ylabel = "greedy_test episode return (a.u.)"
        else:
            ylabel = (
                var_name
                if var_name
                in [
                    "learning_rate",
                    "loss",
                    "explained_variance",
                    "clip_range",
                    "clip_range_vf",
                ]
                else f"{var_name}\n(mean of a window of {agent.n_epochs} updates)"
            )
        # plot and save the figures of previous information
        plt.figure()
        plt.xlabel("Number of network updates")
        plt.ylabel(ylabel)
        plt.plot(x_updates, values)
        plt.tight_layout()
        plt.savefig(f"{path_out}/{var_name}_vs_n_updates.png")

        # write it into a csv
        df_training_updates[var_name] = values
    df_training_updates.to_csv(f"{path_out}/learning_curves_update_data.csv")

    # plot the greedy test as a function of n_experiences
    plt.figure()
    plt.xlabel("Number of training experiences")
    plt.ylabel("greedy_test episode return (a.u.)")
    plt.plot(x_experiences, learning_curves["greedy_test"])
    plt.tight_layout()
    plt.savefig(f"{path_out}/greedy_test_vs_n_experiences.png")

    # write it into a csv
    df_training_experiences["greedy_test"] = learning_curves["greedy_test"]
    df_training_experiences.to_csv(f"{path_out}/learning_curves_experience_data.csv")


def env_monitor_outputs(
    monitored_env: VecEnv | Monitor_custom,
    env: gym.Env,
    path_out: Path,
    verbose: bool = False,
):
    """Output environment monitor records in plot and csv format.

    Parameters
    ----------
    monitored_env : VecEnv | Monitor_custom
        Monitor wrapped gym environment, vectorized or not, where the agent is
        trained.
    env : gym.Env
        Gym environment where the agent is trained.
    path_out : Path
        Folder where monitor plots and csv will be saved.
    verbose : bool, optional
        Activate different loggings, by default False.

    Return
    ------
    total_n_episodes: int
        Total number of episodes explored by the environments.
    """
    if isinstance(monitored_env, VecEnv):
        env_with_monitor = monitored_env.env_is_wrapped(Monitor_custom)
        n_envs = len(env_with_monitor)
        if not all(env_with_monitor):
            raise ValueError("Input environment must be wrapped with `Monitor_custom`.")
    else:
        n_envs = 1
        if not isinstance(monitored_env, Monitor_custom):
            raise ValueError("Input environment must be wrapped with `Monitor_custom`.")

    total_n_episodes = 0
    episode_vars_nd_labels = {
        "episode_times": "Runtime (s)",
        "episode_lengths": "Experiences",
        "episode_returns": "Return (a.u.)",
        "episode_rewards": "Rewards (a.u.)",
        "episode_actions": "Actions",
    }
    plot_vars = ["episode_times", "episode_lengths", "episode_returns"]
    for n_env in range(n_envs):
        episode_records = {}
        # get actions, rewards, return, number of experiences and runtime of
        # each episode
        for var in episode_vars_nd_labels:
            episode_records[var] = call_method_or_attr_of_envs(
                method_name=f"get_{var}", env_to_call=n_env
            )
            # assert all records have the same number of values
            assert len(episode_records[var]) == len(list(episode_records.values())[0])
        total_n_episodes += len(list(episode_records.values())[0])

        # output the information of the episode with best return
        if verbose:
            idx_max_return = np.argmax(episode_records["episode_returns"])
            action_list_max_return = [
                env.action_idx_to_name(action_idx)
                for action_idx in episode_records["episode_actions"][idx_max_return]
            ]
            logging.info(
                f"env {n_env}: Actions list of episode with max return\n"
                f"{action_list_max_return}"
            )
            logging.info(
                f"env {n_env}: Rewards list of episode with max return\n"
                f"{episode_records['episode_rewards'][idx_max_return]}"
            )

        # plot and save the figures of previous information
        for var in plot_vars:
            ylabel = episode_vars_nd_labels[var]
            plt.figure()
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.plot(episode_records[var])
            plt.savefig(f"{path_out}/{ylabel.split(' ')[0]}_vs_episodes_env{n_env}.png")

        # write actions, rewards, returns, n_experiences and runtimes of each
        # episode during training into a csv
        df_training_data = pd.DataFrame(
            data=zip(*episode_records.values(), strict=True),
            columns=episode_vars_nd_labels.values(),
        )
        df_training_data.to_csv(f"{path_out}/training_data_env{n_env}.csv")
    return total_n_episodes


def maskablePPO_train(
    env: gym.Env,
    verbose: bool = False,
    use_masking: bool = True,
    monitor_train: bool = True,
    cfg_path: str = "./config.toml",
    path_out: Path = Path("./models/maskedppo"),
    **env_kwargs,
) -> tuple[MaskablePPO, float, int, float | None]:
    """Train an agent with StableBaselines3 Contrib Maskable PPO algorithm.

    Hiperparameter optimization via Optuna can be selected providing a range of
    values for a/several hiperparameter/s in config.

    Parameters
    ----------
    env : gym.Env
        Gym environment where the agent will be trained.
    hp_optimization : bool
        True to optimize training hiperparameters using Optuna.
    verbose : bool, optional
        Activate different loggings, by default False.
    use_masking : bool, optional
        Whether to employ action masking or not, by default True.
    monitor_train : bool, optional
        Monitor training and save monitorization data and plots. It will be
        saved in the same folder than the trained agent.
    cfg_path : str, optional
        Path to config file, by default "./config.toml".
    path_out : Path, optional
        Path where the trained agent and monitor plots/csv will be saved, by
        default Path("./models/maskedppo").
    **env_kwargs
        Inputs of input `env`, if necessary.

    Return
    ------
    MaskablePPO
        Trained agent.
    train_time : float
        Training time of the returned agent.
    n_explored_episodes : int
        Number of episodes explored by the environment during training.
    optuna_time : float | None
        Time of hyperparameter optimization.

    Warnings
    --------
    * Names of hyperparameters in config file have to be the same than
      MaskablePPO inputs. If not, they will be not considered.
    * Training graphs are only given for the agent trained with the optimal
      hiperparameters, found by Optuna.
    In order to not modify Stable-Baselines3, info output of training steps will
    not be retrieved. However, during training:
    * Missing uv data can be tracked setting logging level to DEBUG.

    References
    ----------
    .. [1] https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    .. [2] https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
    .. [3] https://stable-baselines3.readthedocs.io/en/master/common/monitor.html
    .. [4] https://github.com/optuna/optuna?tab=readme-ov-file#basic-concepts
    .. [5] https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
    .. [6] https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    .. [7] https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study
    .. [8] https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results
    .. [9] https://optuna.readthedocs.io/en/stable/faq.html#how-are-exceptions-from-trials-handled
    .. [10] https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
    """
    ppo_cfg = load_conf(cfg_path)["agent"]["PPO"]

    # store params not to be optimized for training
    int_params = ["n_steps", "batch_size", "n_epochs"]
    float_params = ["learning_rate", "gamma", "gae_lambda", "ent_coef"]
    params2opt = [key for key, value in ppo_cfg.items() if isinstance(value, list)]
    params4training = {
        hp: ppo_cfg[hp] for hp in int_params + float_params if hp not in params2opt
    }

    # ------- HP OPTIMIZATION -------
    if params2opt:
        logging.info(f"Optuna optimization selected for {params2opt}")

        # Define an objective function to be minimized.
        def objective(trial: optuna.Trial) -> float:
            objective_params4training = params4training.copy()
            # set floats to tune with optuna, determining their min, max and log/step
            MaskablePPO_hp2tune = dict()
            for param in params2opt:
                if param in float_params:
                    log, float_step = check_bool_or_float(ppo_cfg[param][2])
                    MaskablePPO_hp2tune[param] = trial.suggest_float(
                        param,
                        ppo_cfg[param][0],
                        ppo_cfg[param][1],
                        log=log,
                        step=float_step,
                    )
                elif param in int_params:
                    log, int_step = check_bool_or_int(ppo_cfg[param][2])
                    MaskablePPO_hp2tune[param] = trial.suggest_int(
                        param,
                        ppo_cfg[param][0],
                        ppo_cfg[param][1],
                        log=log,
                        step=int_step,
                    )
                else:
                    raise NotImplementedError(
                        f"{param} not scripted for Optuna optimization."
                    )

            # DISCLAIMER: reset the AGENT environment to ensure an initialized
            # environment seed, as it can be dragged from previous trial.
            if ppo_cfg["n_envs"] == 0:
                agent_env = env(**env_kwargs)
                agent_env.reset(seed=ppo_cfg["seed"])
            else:
                # SB3 docs [10]: the reset() method doesn’t take any parameter.
                # If you want to seed the pseudo-random generator or pass
                # options, you should call vec_env.seed(seed=seed) /
                # vec_env.set_options(options) and obs = vec_env.reset()
                # afterward (seed and options are discarded after each call to
                # reset()).
                agent_env = make_vec_env_custom(
                    env,
                    env_kwargs=env_kwargs,
                    seed=ppo_cfg["seed"],
                    n_envs=ppo_cfg["n_envs"],
                    vec_env_cls=(
                        SubprocVecEnv if ppo_cfg["env_multiprocess"] else DummyVecEnv
                    ),
                )
                agent_env.seed(seed=ppo_cfg["seed"])
                agent_env.reset()

            # merge given value and trial parameters
            trial_params = objective_params4training | MaskablePPO_hp2tune
            # set the agent
            agent = MaskablePPO(
                policy="MlpPolicy",
                env=agent_env,
                verbose=0,
                seed=ppo_cfg["seed"],
                _init_setup_model=True,
                **trial_params,
            )
            # train it
            agent.learn(
                total_timesteps=ppo_cfg["total_timesteps"],
                progress_bar=False,
                use_masking=use_masking,
            )

            # DISCLAIMER: reset the EPISODE environment to ensure an initialized
            # environment, as it can be dragged from previous trial
            episode_env = env(**env_kwargs)
            episode_env.reset(seed=ppo_cfg["seed"])
            # obtain the value to optimize
            _, _, _, _, _, episode_rewards, _ = maskablePPO_episode(
                start_state=env_kwargs["start_env"],
                agent=agent,
                env=episode_env,
                deterministic=True,
                use_masking=use_masking,
            )

            # close previous envs
            agent_env.close()
            episode_env.close()

            # return the objective value to optimize
            return sum(episode_rewards)

        with timer(tag="optuna_time") as optuna_time:
            # create a new study and set the maximization of the objective
            sampler = optuna.samplers.TPESampler(seed=ppo_cfg["seed_optuna"])  # [8]
            study = optuna.create_study(sampler=sampler, direction="maximize")  # [5]

        # invoke optimization of the objective function
        # WARNING:`n_jobs` must be 1 in order to not lose squential dependent
        # variables such as `terminated`, `_visited_actions_memory`, etc.
        study.optimize(
            objective, n_trials=ppo_cfg["n_trials"], n_jobs=1, catch=ValueError
        )  # [6], [9]

        # ------- PRINT OPTIMIZATION RESULTS [7] -------
        logging.info(
            "Number of trials fail / complete: ",
            study.trials_dataframe().groupby("state").count()["number"],
        )
        logging.info("Best trial:")
        best_agent = study.best_trial  # (.trials for all of them)
        logging.info(f"  Value: {best_agent.value}")
        logging.info("  Params: ")
        for key, value in best_agent.params.items():
            logging.info(f"    {key}: {value}")
        logging.info("  User attrs:")
        for key, value in best_agent.user_attrs.items():
            logging.info(f"    {key}: {value}")

    # ------- TRAIN WITH INDICATED/OPTIMIZED HP -------
    # wrap and initialize the environment
    if ppo_cfg["n_envs"] == 0:
        # ... to be monitored
        train_env = env(**env_kwargs)
        train_env.reset(seed=ppo_cfg["seed"])
        monitored_env = Monitor_custom(train_env) if monitor_train else train_env
        monitored_env.reset(seed=ppo_cfg["seed"])
    else:
        # ...to be vectorized and monitored
        monitored_env = make_vec_env_custom(
            env,
            env_kwargs=env_kwargs,
            seed=ppo_cfg["seed"],
            n_envs=ppo_cfg["n_envs"],
            vec_env_cls=SubprocVecEnv if ppo_cfg["env_multiprocess"] else DummyVecEnv,
        )
        monitored_env.seed(seed=ppo_cfg["seed"])  # [10]
        monitored_env.reset()

    # select final training hiperparameters
    train_params = (
        params4training | best_agent.params if params2opt else params4training
    )

    with timer(tag="train_time") as train_time:
        # set the agent
        agent = MaskablePPO_custom(
            policy="MlpPolicy",
            env=monitored_env,
            verbose=2 if verbose else 1,
            seed=ppo_cfg["seed"],
            _init_setup_model=True,
            **train_params,
        )
        # train it and capture training outputs
        with Capturing() as learn_outputs:
            agent.learn(
                total_timesteps=ppo_cfg["total_timesteps"],
                progress_bar=verbose,
                use_masking=use_masking,
                greedy_check_interval=ppo_cfg["greedy_check_interval"],
                greedy_env=env(**env_kwargs),
            )
    # save it
    agent.save(f"{path_out}/{path_out.stem}")

    if monitor_train:
        n_explored_episodes = env_monitor_outputs(monitored_env, env, path_out, verbose)
        agent_training_outputs(learn_outputs, agent, path_out)

    monitored_env.close()
    return (
        agent,
        train_time(),
        n_explored_episodes,
        optuna_time() if params2opt else None,
    )
