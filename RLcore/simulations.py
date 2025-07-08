import logging
from pathlib import Path

import gymnasium as gym
from agent_predict import maskablePPO_episode
from agent_train import maskablePPO_train
from sb3_contrib import MaskablePPO


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
