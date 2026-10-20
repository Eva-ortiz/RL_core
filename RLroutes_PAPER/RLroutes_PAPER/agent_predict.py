import logging
from collections import deque
from typing import Any

import gymnasium as gym
from sb3_contrib import MaskablePPO

from RLroutes_PAPER.environment import Reward, State, State_norm


def maskablePPO_step_prediction(
    state: State | State_norm,
    agent: MaskablePPO,
    env: gym.Env,
    deterministic: bool = True,
    use_masking: bool = True,
) -> tuple[gym.Env, State_norm, Reward, bool, bool, dict[str, Any]]:
    """Perform a prediction from input state with provided agent.

    Parameters
    ----------
    state: State
        State from where to make the prediction.
    agent: MaskablePPO
        Agent to employ during the episode.
    env : gym.Env
        Initialized environment object to run the input agent.
    deterministic : bool, optional
        Whether to return or not deterministic actions.
    use_masking : bool, optional
        Whether to employ action masking or not, by default True.

    Returns
    -------
    env : gym.Env
        Environment object with performed step.
    norm_next_state : State_norm
        Next agent observation, normalized.
    reward : Reward
        Reward for taking input action.
    terminated : bool
        If the environment has terminated due to input action.
    truncated : bool
        If the environment has truncated due to input action.
    info : dict[str, Any]
        Additional information from the environment about the step.

    References
    ----------
    .. [1] https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO.predict
    .. [2] https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    """
    if isinstance(state, State):
        norm_state = env._normalize_state_values_single(state[env.single_state_cols])
        if env.global_obs:
            norm_state = env._normalize_state_values_global(
                all_buoy_keys=env._action_buoykey_dict.values(),
                state_global=state[env.global_state_cols],
                norm_single_state=norm_state,
                max_capture=env.conf["max_single_capture"],
            )
    elif isinstance(state, State_norm):
        norm_state = state
    else:
        raise ValueError("Wrong type of `state` input.")

    action_masks = env.action_masks() if use_masking else None
    action, _ = agent.predict(
        norm_state, deterministic=deterministic, action_masks=action_masks
    )  # [1]

    # we extract the action number with `.item()` as we are predicting from
    # just one observation
    norm_next_state, reward, terminated, truncated, info = env.step(
        action.item()
    )  # [2]

    return env, norm_next_state, reward, terminated, truncated, info


def maskablePPO_episode(
    start_state: State | State_norm,
    agent: MaskablePPO,
    env: gym.Env,
    deterministic: bool = True,
    use_masking: bool = True,
) -> tuple[deque, deque, deque, deque, deque, deque, list[dict[str, Any]]]:
    """Simulate an episode in `env` from a `start_state` with trained `agent`.

    Maskable PPO implemented with StableBaselines3 Contrib.

    Parameters
    ----------
    start_state: State | State_norm
        State to start the episode from.
    agent: MaskablePPO
        Agent to employ during the episode.
    env : gym.Env
        Initialized environment object to run the input agent.
    deterministic : bool, optional
        Whether or not to return deterministic actions.
    use_masking : bool, optional
        Whether to employ action masking or not, by default True.

    Returns
    -------
    episode_buoys : deque
        Deque with the buoy key of each step.
    episode_lats : deque
        Deque with the latitude of each step.
    episode_lons : deque
        Deque with the longitude of each step.
    episode_dates : deque
        Deque with the date of each step.
    episode_times : deque
        Deque with the time of each step.
    episode_rewards : deque
        Deque with the reward of each step.
    info_list : list[dict[str, Any]]
        Additional information about taken steps.

    Notes
    -----
    Stable-Baselines3 evaluation helper could be used instead for simpler
    evaluations. Refer to [4] for more information.

    References
    ----------
    .. [1] https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO.predict
    .. [2] https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    .. [3] https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
    .. [4] https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html#eval
    """
    info_list = []

    # initialize terminated, truncated
    terminated, truncated = False, False

    # initialize storage of environment components and rewards
    episode_buoys = deque([env.current_env[env.buoy_col]])
    episode_lats = deque([env.current_env[env.lat_col]])
    episode_lons = deque([env.current_env[env.lon_col]])
    episode_dates = deque([env.current_env[env.date_col]])
    episode_times = deque([env.current_env[env.time_col]])
    episode_rewards = deque([0])

    # generate the episode
    state = start_state
    while not terminated and not truncated:
        env, state, reward, terminated, truncated, info = maskablePPO_step_prediction(
            state=state,
            agent=agent,
            env=env,
            deterministic=deterministic,
            use_masking=use_masking,
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

    # output relevant info
    logging.info(f"Episode return: {sum(episode_rewards)}")

    return (
        episode_buoys,
        episode_lats,
        episode_lons,
        episode_dates,
        episode_times,
        episode_rewards,
        info_list,
    )
