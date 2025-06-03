import copy
import logging
import warnings
from collections import deque, namedtuple
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from basics import agent
from environment import ENV, Action, Reward, Setup_mode, State, State_norm
from plots import learning_curve
from utils import HTC_units_label

from RLcore.utils import str_to_tuple_or_list

transition = namedtuple(
    "transition", ("state_norm", "action_idx", "reward", "next_state_norm")
)
sarsa_transition = namedtuple(
    "transition",
    ("state_norm", "action_idx", "reward", "next_state_norm", "next_action_idx"),
)


class ReplayMemory:
    """Store experiences (s,a,r,s') to train Reinforcement Learning algorithm.

    Tuned code from Pytorch. [1]

    References
    ----------
    .. [1] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
    .. [2] https://docs.python.org/3/library/collections.html#collections.deque
    """

    def __init__(self, capacity, agent, **kwargs):
        experiences, self.visited_states, self.reached_states, _, _ = (
            agent._experience_generation(**kwargs)
        )
        self.memory = deque(experiences, maxlen=capacity)

    def push(self, experiences: list[transition]):
        """Save input experiences and visited states.

        When `capacity` is reached, `deque` iterator automatically remove older
        elements when new ones are appended. [2]
        """
        for experience in experiences:
            self.memory.append(experience)
            self.visited_states.add(experience.state)

    def sample(self, batch_size, random_rng):
        """Sample `batch_size` stored experiences.

        Use provided random rng for it.
        """
        return random_rng.sample(self.memory, batch_size)

    def __len__(self):
        """Memory len."""
        return len(self.memory)


class QNN(nn.Module):
    """Neural Network for estimating all the action values from a given state.

    References
    ----------
    .. [1] https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, hidden_neur: int):
        """Init function of the action value network.

        Parameters
        ----------
        in_dim : int
            Number of inputs of the action value network.
            It will be equal to the number of the features of a state.
        out_dim : int
            Number of outputs of the action value network.
            It will be equal to the number of actions or q values for a single
            state.
        hidden_layers : int
            Number of desired hidden layers.
        hidden_neur : int
            Number of neurons of each one of the hidden layers.
        """
        super().__init__()

        # add input layer
        layers = [nn.Linear(in_dim, hidden_neur), nn.SELU()]

        # Add hidden layers
        for _ in range(hidden_layers):
            # Append the hidden layer and its corresponding activation function
            layers.append(nn.Linear(hidden_neur, hidden_neur))
            layers.append(nn.SELU())

        # add the output layer
        layers.append(nn.Linear(hidden_neur, out_dim))

        # set the network
        self.selu_stack = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the network."""
        y = self.selu_stack(x)
        return y

    def copy_weights(self, net_to_copy):
        """Copy the weights of a give network.

        References
        ----------
        .. [1] Fundations of Deep Reinforcement Learning; Laura Graesser and Wah
            Loon Keng
        """
        self.load_state_dict(net_to_copy.state_dict())


class DRL_agent(agent):
    """Deep Reinforcement Learning agent.

    References
    ----------
    .. [1] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(
        self,
        algorithm: Literal["Sarsa", "Q-learning", "double_Q-learning"],
        actions: np.typing.ArrayLike,
        save_folder: str = "DRL_results",
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Set Base Deep Reinforcement Learning algorithm for agents.

        Parameters
        ----------
        algorithm : Literal["Sarsa", "Q-learning", "double_Q-learning"]
            Determine the Reinforcement Learning algorithm to use. Implemented
            algorithm are:
                * Sarsa
                * Q-learning
                * double Q-learning
        states : Iterable[str, Number]
            Collection of all possible states of the problem.
        actions : np.typing.ArrayLike
            Collection of all possible actions of the problem.
        save_folder : str, optional
            Default name of the save folder for the outputs of the algorithm.
            By default, "DRL_outputs"
        seed : int | None, optional
            Seed for the pseudo random generators
        verbose : bool, optional
            Select verbose though the algorithm.
            By default, False

        Other Parameters
        ----------------
        **kwargs
            hidden_layers : int, optional
                Number of hidden layers of the action values neural network.
                By default, 2
            hidden_neur : int, optional
                Number of neurons of each one of the hidden layers of the action
                values neural network.
                By default, 16
        """
        # inherit from parent class
        super().__init__(algorithm, actions, seed, verbose)

        # outputs save folder
        self.save_folder = save_folder

        # store more relevant attributes for the algorithm
        self.hidden_layers = kwargs.get("hidden_layers", 2)
        self.hidden_neur = kwargs.get("hidden_neur", 16)

    def greedy_simulation(
        self,
        q_net: QNN,
        environment: ENV,
        max_steps: int,
        device: Literal["cuda", "mps", "cpu"],
        reset_options: dict[str, Any] | None = None,
    ) -> tuple[float, Reward, Reward, State_norm, str]:
        """Greedy simulation with current Q network and reset environment.

        Parameters
        ----------
        q_net : QNN
            Network for the prediction of all action-state values for a given
            state.
        environment : ENV
            Environment object of the problem.
        max_steps : int
            Maximum number of steps of the simulation. If `end_episode` reached,
            previously stop the simulation.
        device : Literal["cuda", "mps", "cpu"], optional
            Currently used device for training. Used as an `act` method input.
        reset_options : dict[str, Any] | None, optional
            Additional information to specify how the environment is reset. By
            default, None.

        Returns
        -------
        overall_return : float
            Value of the return for the greedy simulation.
        last_reward : Reward
            Value of the last reward of the simulation.
        best_reward : Reward
            Value of the best reward seen during the simulation.
        last_state_norm : State_norm
            Last visited state (normalized). Useful to continue the trajectory
            of (s, a, r, s') generated.
        last_action : str
            Label of the last action performed. For informative purposes.

        Warnings
        --------
        It is recommended that input environment is an independent copy of the
        environment used for the rest algorithm, as it will be reset.

        See Also
        --------
        environment.py : where environment.reset() method is defined.
        MaskablePPO_custom.greedy_simulation : where an example implementation
        of `greedy_simulation` can be seen for inspo.
        """
        # initialize some parameters
        step = 1
        terminated, truncated = False, False
        overall_return = 0
        best_reward = -np.inf

        # reset the environment for this simulation and select start state
        state_norm, _ = environment.reset(seed=self.seed, options=reset_options)

        # generate the simulation
        while step < max_steps or (not terminated and not truncated):
            # get action with greedy policy, as we want to evaluate the
            # optimality of the `q_net`
            action_idx, _, action_label = self._act(
                "greedy",
                state_norm,
                q_net=q_net,
                device=device,
            )

            # observe response of the environment
            state_norm, reward, terminated, truncated, _ = environment.step(action_idx)

            # store best reward of the simulation
            if reward > best_reward:
                best_reward = reward

            # store the return of the simulation
            overall_return += reward

            if terminated or truncated:
                reason_str = "TERMINATED" if terminated else "TRUNCATED"
                warnings.warn(
                    f"Simulation finalized at step {step} due to {reason_str} episode.",
                    stacklevel=1,
                )
                break

            step += 1

        # output the return, last reward value, best reward value and last state
        # (normalized) and last action label
        return overall_return, reward, best_reward, state_norm, action_label

    def _experience_generation(
        self,
        n_experiences: int,
        q_net: QNN,
        environment: ENV,
        device: Literal["cuda", "mps", "cpu"],
        epsilon: float,
        initial_state: State | State_norm,
        initial_action: Action | None,
        follow_next_action: bool = False,
        decorrelated: bool = False,
        **kwargs,
    ) -> tuple[
        list[namedtuple],
        tuple[State_norm],
        tuple[State_norm],
        State_norm,
        Action | None,
    ]:
        """Generate experiences consisting of states, actions and rewards.

        Used to train the agent. The generated values depends of the target
        algorithm.

        Parameters
        ----------
        n_experiences : int
            Number of experiences to generate.
        q_net : QNN
            Network for the prediction of all action-state values for a given
            state.
        environment : ENV
            Environment object of the problem.
        device : Literal["cuda", "mps", "cpu"]
            Currently used device for training.
        epsilon : float
            Value of epsilon in epsilon greedy policy. With higher
            epsilon, more exploratory behaviour of the policy.
        initial_state : State | State_norm
            State to initialize the generation of experiences from.
        initial_action : Action | None
            First action index. If None, select an action according to
            epsilon_greedy behaviour policy.
        follow_next_action : bool, optional
            Select to store and follow a' along all experiencies generation.
            By default, False.
        decorrelated : bool, optional
            Select if experience samples are decorrelated. If True, they will.
            By default, False.

        ** kwargs
            reset_options : dict, optional
                Additional information to specify how the environment is reset
                (depending on the specific environment). By default, None.

        Returns
        -------
        experiences : list[namedtuple]
            List which contains each one of the experiences, composed by
            (state_norm, action_idx, reward, next_state_norm).
            If follow_next_action, also include next_action_idx.
        visited_states : tuple[State_norm]
            Tuple of visited states (normalized).
        reached_states : tuple[State_norm]
            Tuple of reached states (normalized), useful when the reward is
            computed as a function of the next state.
        last_next_state : State_norm
            Last state s' (normalized) visited. Useful to continue the
            trajectory of (s, a, r, s') generated.
        last_next_action : Action | None
            Last action a' performed. Useful to continue the sequence of
            generated experiences.

        Warnings
        --------
        * `State` and `State_norm` dtypes must be different or initial state
          normalization will not be applied.
        * Pay A LOT of attention to environment state track:
          `step` does not have the state as input, so we have to be very careful
          with how we determine the state we want to make the step from.

        See Also
        --------
        environment
            Where kwargs such as `reset_options` will be applied as input. Its
            methods are also crucial for the correct functioning of this
            experience generation.
        MaskablePPO.collect_rollouts
            TODO : Method from `MaskablePPO` algorithm, reference to use
            vectorized environments.

        Notes
        -----
        * End of episode can be reached while experience generation. In this
          case, a restart of the environment will be done.
        * Decorrelated transitions are given through random sampling of states.
          Several authors recommend this practice.
        * We will skip transitions that starts at the terminal state, defined by
          the environment (see `environment._setup`).
        """
        # if input initial state is a terminal state, require a new input
        if environment._termination(initial_state):
            raise ValueError("Initial state provided is a terminal state.")

        # if input initial action has more than one element, require the user to
        # select just one
        if initial_action is not None and len(initial_action) > 1:
            raise ValueError(
                "More than one initial action provided, please select just one."
            )

        # obtain normalized state from initial state
        state_norm = (
            initial_state
            if isinstance(initial_state, State_norm)
            else environment._normalize_state_values(initial_state)
        )
        # set action as initial action
        action_idx = initial_action

        # store generated experiences
        experiences = []
        visited_states = set()
        reached_states = set()

        # TODO : obtain actions with vectorized environments, for ref see
        # MaskablePPO.collect_rollouts
        for _ in range(n_experiences):
            # if decorrelated selected, randomly select next state and set the
            # environment to this state
            if decorrelated:
                # Reset the environment setting start env/state to None and
                # current env to a random state. In addition, reset some
                # counters.
                state_norm = environment._setup(
                    mode=Setup_mode.INIT, start_env="random"
                )

            # store visited states
            visited_states.add(state_norm)

            # if action has not been provided as input, choose action with
            # behaviour policy
            if action_idx is None:
                action_idx, _, _ = self._act(
                    "epsilon_greedy",
                    state_norm,
                    q_net=q_net,
                    device=device,
                    epsilon=epsilon,
                )

            # observe response of the environment
            next_state_norm, reward, terminated, truncated, _ = environment.step(
                action_idx
            )
            # store reached states
            reached_states.add(next_state_norm)

            # store the transition
            if not follow_next_action:
                experiences.append(
                    transition(state_norm, action_idx, reward, next_state_norm)
                )

            # store sarsa transition if track_next_action is selected
            else:
                # perform next action too
                next_action_idx, _, _ = self._act(
                    "epsilon_greedy",
                    next_state_norm,
                    q_net=q_net,
                    device=device,
                    epsilon=epsilon,
                )

                # store next action in transition
                experiences.append(
                    sarsa_transition(
                        state_norm,
                        action_idx,
                        reward,
                        next_state_norm,
                        next_action_idx,
                    )
                )

            # consider the end of the episode or continue from next state
            state_norm, _ = (
                environment.reset(options=kwargs.get("reset_options"))
                if terminated or truncated
                else next_state_norm
            )

            # If follow_next_action is selected, force reset action to None if
            # terminated or truncated has been reached. Otherwise, set action to
            # next_action.
            if follow_next_action:
                action_idx = None if terminated or truncated else next_action_idx

            # set action to None once its input has been employed, so we do not
            # get stuck in the initial action for follow_next_action = False
            else:
                action_idx = None

        # output stored experiences, visited and reached states, in addition to
        # the final visited state and next action
        return (
            experiences,
            visited_states,
            reached_states,
            state_norm,
            action_idx,
        )

    def _act(
        self,
        mode: Literal["greedy", "epsilon_greedy"],
        state_norm: State_norm,
        q_net: QNN,
        device: Literal["cuda", "mps", "cpu"],
        **kwargs,
    ) -> tuple[int, float, str]:
        """Return the action that the agent takes given an state.

        In other words, this is the application of the policy.

        Parameters
        ----------
        mode : Literal["greedy", "epsilon_greedy"]
            Mode of acting.
            We can select:
                * "greedy" actions
                * "epsilon_greedy" actions
        state_norm: State_norm
            Normalized state where the agent currently is.
        q_net : QNN
            Network for the prediction of all action-state values for a given
            state.
        device : Literal["cuda", "mps", "cpu"]
            Currently used device for training.

        **kwargs
            epsilon : float
                Value of epsilon in epsilon greedy policy. With higher
                epsilon, more exploratory behaviour of the policy.

        Returns
        -------
        action_idx, q_value, action_label : tuple[int, float, str]
            Index, value and label of the action taken by the agent.

        Warnings
        --------
        It is understood that the order of the output q_values from the `q_net`
        is the same that the order of the actions in `self.actions`. Else,
        returns will be inconsistent with the desired functionality.

        References
        ----------
        ..[1] https://pytorch.org/docs/stable/generated/torch.max.html#torch.max
        """
        # Obtain the action-state values for all actions from input `state`
        # Additionally, execute the forward pass at the same device we are using for
        # training to avoid a Pytorch `RuntimeError`
        # OUTDATED : It is necessary to set input as float32 so Pythorch does
        # not return us a `RuntimeError` due dtypes
        q_values = q_net.forward(torch.from_numpy(state_norm).to(device))

        # -------- BASIC CHECKS --------
        # check if the number of outputs are the same than the number of actions
        if len(self.actions) != q_values.shape[0]:
            raise ValueError(
                "Outputs of the `q_net` should correspond to the actions "
                "stored in `self.actions`, even respecting the order."
            )

        # -------- ACTION SELECTION --------
        # epsilon greedy behaviour
        if mode == "epsilon_greedy":
            try:
                epsilon = kwargs["epsilon"]
            except KeyError as err:
                raise ValueError(
                    "Epsilon of epsilon greedy policy not provided."
                ) from err

            rand = self.random_rng.uniform(0, 1)

        # greedy behaviour
        elif mode == "greedy":
            # always get a random number above epsilon, so random choice is
            # never taken
            epsilon = 0
            rand = 1

        # select the action depending on a random number and epsilon value
        if epsilon > rand:
            action_idx = self.random_rng.randint(0, len(self.actions) - 1)

        # select the action with a greedy policy
        else:
            # store the max action value
            max_val = torch.max(q_values)

            # store the max actions indexes
            max_action_idxs = [
                idx for idx, q_value in enumerate(q_values) if q_value == max_val
            ]

            # select randomly the action between actions which presents the
            # maximum value (so if there is a tie, `torch.max` do not take
            # always the first action) [2]
            action_idx = self.random_rng.choice(max_action_idxs)

        # get action value and label with selected index
        action_val = q_values[action_idx]
        action_label = self.actions[action_idx]

        # check if the action seected is the maximum value action with greedy
        # behaviour
        if mode == "greedy":
            assert max_val == action_val

        return action_idx, action_val, action_label

    def _check_train_inputs(
        self,
        epsilon: float,
        lr: float,
        discount_rate: float,
        plot_learning_curves: bool,
        reward_curve_mode: list[str],
        reward_curve_steps_per_point: int,
    ):
        """Check of `.train` inputs."""
        if epsilon > 1 or epsilon < 0:
            raise ValueError("Epsilon must be a number between 0 and 1.")
        if lr > 1 or lr < 0:
            raise ValueError("Learnig rate must be a number between 0 and 1.")
        if discount_rate > 1 or discount_rate < 0:
            raise ValueError("Discount rate must be a number between 0 and 1.")
        if plot_learning_curves and reward_curve_steps_per_point is None:
            warnings.warn(
                """In order to plot reward curves,
                `reward_curve_steps_per_point` must be other than None.""",
                stacklevel=1,
            )
        if not set(reward_curve_mode) <= {"return", "last_reward", "best_reward"}:
            raise ValueError(
                """Invalid reward curve mode. Please, select 'return',
                'last_reward' or 'best_reward'."""
            )

    def train(
        self,
        device: Literal["cuda", "mps", "cpu"],
        environment: gym.Env,
        discount_rate: float = 0.99,
        lr: float = 0.1,
        epsilon: float = 1.0,
        batch_size: int = 64,
        max_steps: int = np.inf,
        tol_loss: float = 0.0,
        plot_learning_curves: bool = True,
        save_q_net: bool = True,
        **kwargs,
    ):
        """Train a NN to predict action-state values from input states.

        Parameters
        ----------
        device : Literal["cuda", "mps", "cpu"]
            Currently used device for training. Used as an `act` method input.
        environment : environment
            Environment object of the problem.
        discount_rate : float, optional
            Discount rate factor for Reinforcement Learning algorithm.
            By default, 0.99
        lr : float, optional
            Value of the learning rate. By default, 0.1
        epsilon : float, optional
            Initial value of epsilon for epsilon greedy policies.
            By default, 1.0
        batch_size : int, optional
            Size of the batch for training the neural network for the prediction
            of the action-state values. By default, 64
        max_steps : int, optional
            Maximum number of steps to iterate policy evaluation.
            By default, np.inf
        tol_loss : float, optional
            Tolerance to consider action values have converged. By default, 0.0
        plot_learning_curves : bool, optional
            Select to plot learning curves or not. By default, True
        save_q_net : bool, optional
            Select to save the action-state values network. By default, True.

        Other Parameters
        ----------------
        **kwargs
            reduce_eps : float, optional
                Amount to reduce epsilon at each iteration. By default, 1e-3
            min_eps : float, optional
                Minimum value of epsilon. By default, 0.0

            reduce_perc_lr : float, optional
                Percentage to reduce learning rate at each iteration.
                By default, 0.0001
            min_lr : float, optional
                Minimum value of learning rate. By default, 0.0

            reward_curve_mode : list[str], optional
                List with the selection of the rewards to record at `reward_curve`:
                    * return : plot the return of the start state for each
                        simulation.
                    * last_reward : plot the reward of the last step of each
                        simulation.
                    * best_reward : plot the best reward seen during all the
                      training.
                By default, ["return", "last_reward", "best_reward"]
            reward_curve_steps_per_point : Optional[int], optional
                Select the number of steps for each one of the simulated
                episodes created to plot each point of the reward curve.
                If None, DO NOT RECORD any reward learning curve.
                By default, 30.

            episode_start_state : [Literal["rand", "initial"], str], optional
                State to initialize the episodes from.
                If "rand", create a random initial state each time.
                If "initial", start from environment start state.
                By default, "initial".

            decorrelated : bool, optional
                Select if batch samples are decorrelated. If True, they will be
                decorrelated. By default, False.
            step_count : bool, optional
                Select to have a steps counter. Specially useful for those
                algorithms which do not have a natural terminal state, so it is
                implemented as an episode length.
                By default, False, so step count is inactive.

            debug_counter : int, optional
                Select every how many counts logging debug will be given.
                By default, 1.

            seed : int, optional
                Seed number for random and torch modules. If None, do not fix
                any seed. By default, None.

        Warns
        -----
        * If `plot_learning_curves` is selected but
          `reward_curve_steps_per_point` is None.
        * If `reward_curve_steps_per_point` is set to other than None.

        Notes
        -----
        * Currently, one pair action-state is being updated for each individual
          sample of the batch.
        * Notice that `max_steps` is not related to the number of steps per
          episode. For this, we have environment step count. In addition,
          the number of samples to update the network at each step is determined
          by batch parameter.
          Meanwhile, tabular method is trained for a number of episodes composed
          by its number of steps.

        References
        ----------
        .. [1] https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        .. [2] https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
        .. [3] https://stackoverflow.com/questions/48324152/how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no-lr-sched
        """
        # fix seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # info
        logging.info(f"Training the agent with {self.algorithm} algorithm...")

        # ------ KWARGS ------
        reduce_eps = kwargs.get("reduce_eps", 1e-3)
        min_eps = kwargs.get("min_eps", 0.0)

        reduce_perc_lr = kwargs.get("reduce_perc_lr", 0.0001)
        min_lr = kwargs.get("min_lr", 0.0)

        reward_curve_mode = kwargs.get(
            "reward_curve_mode", ["return", "last_reward", "best_reward"]
        )
        reward_curve_steps_per_point = kwargs.get("reward_curve_steps_per_point", 30)

        episode_start_state = kwargs.get("episode_start_state", "initial")

        decorrelated = kwargs.get("decorrelated", False)
        step_count = kwargs.get("step_count", False)

        debug_counter = kwargs.get("debug_counter", 1)

        # basic checks of input values
        self._check_train_inputs(
            epsilon,
            lr,
            discount_rate,
            plot_learning_curves,
            reward_curve_mode,
            reward_curve_steps_per_point,
        )

        # -------------------------------------------------------------------------
        # Step 0: define the NN of the Q function
        # -------------------------------------------------------------------------
        # Input dim is the number of aspects that define our state.
        # DISCLAIMER : intended for 1D observation spaces
        in_dim = environment.observation_space.shape[0]

        # Output dim will be given by the number of possible actions for each
        # state, i. e., number of possible actions.
        out_dim = environment.action_space.shape[0]

        # create the neural network
        q_net = QNN(
            in_dim,
            out_dim,
            hidden_layers=self.hidden_layers,
            hidden_neur=self.hidden_neur,
        )
        # and select its training hardware
        q_net = q_net.to(device)

        # set the optimizer
        optimizer = optim.Adam(q_net.parameters(), lr=lr)

        # set the loss
        loss_L1 = nn.L1Loss()

        # -------------------------------------------------------------------------
        # Step 1: perform the training of the system
        # -------------------------------------------------------------------------
        # ------ INITIALIZATION ------
        # initialize epsilon, amount of loss and storage of best reward seen
        # along all the training
        step = 1
        loss = np.inf
        training_best_reward = -np.inf

        # initialize learning curves
        loss_curve = learning_curve()
        reward_curves = []
        if reward_curve_steps_per_point is not None:
            for _ in reward_curve_mode:
                reward_curves.append(learning_curve())

            warnings.warn(
                """In order to save `reward_curves`, an additional simulation at
                each step of the algorithm is made. Please, consider if this
                type of plot is necessary. If not, set
                `reward_curve_steps_per_point` to None.""",
                stacklevel=1,
            )
        known_states_list = []

        # store visited states
        visited_states = set()

        # define episode start with reset method
        episode_start_state = environment.reset(episode_start_state)

        # initialize batch state
        initial_batch_state = episode_start_state

        # select to store a' only for Sarsa algorithm
        follow_next_action = self.algorithm == "Sarsa"
        initial_batch_action = None

        # ------ ALGORITHM ------
        # loop during a determined number of steps or until convergence
        while step <= max_steps and tol_loss < loss:
            # output step info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"Computing step number {step}...")

            # -------------------------------------------------------------------------
            # Step 1.1: generate for each step the experiences for training and
            # store them in a batch
            # -------------------------------------------------------------------------
            (
                batch,
                batch_visited_states,
                _,
                last_batch_state,
                last_batch_action,
            ) = self._experience_generation(
                batch_size,
                q_net,
                environment,
                device,
                epsilon,
                initial_batch_state,
                initial_batch_action,
                follow_next_action,
                decorrelated,
                step_count=step_count,
            )
            initial_batch_state = last_batch_state
            initial_batch_action = last_batch_action

            # store unique visited states with the usage of set
            for state in batch_visited_states:
                visited_states.add(state)

            # -------------------------------------------------------------------------
            # Step 1.2: use the batch experiences to get several pairs
            # estimation and target q values
            # -------------------------------------------------------------------------
            batch_estimations = []
            batch_targets = []
            for experience in batch:
                # Transform state form label to list[int]
                state_list = str_to_tuple_or_list(experience.state, to="list")

                # obtain ALL the q value estimations of the net for state
                # It is necessary to set input as float32 so Pythorch does not
                # return us a `RuntimeError` due dtypes
                # Additionally, execute the forward pass at the same device we
                # are using for training to avoid a Pytorch `RuntimeError`
                estimated_q_values = q_net.forward(
                    torch.from_numpy(np.array(state_list, dtype=np.float32)).to(device)
                )
                batch_estimations.append(estimated_q_values[experience.action_idx])

                # do not follow the gradient for obtained target values
                with torch.no_grad():
                    # take the q value for the next action-state pair with
                    # target policy (Sarsa, Q-learning)
                    if self.algorithm == "Sarsa":
                        # retrieve from experience the next action taken with
                        # behaviour policy and compute its q_value
                        # 1- obtain the action-state values for all actions from
                        # `next_state`
                        next_state_list = str_to_tuple_or_list(
                            experience.next_state, to="list"
                        )
                        next_q_values = q_net.forward(
                            torch.from_numpy(
                                np.array(next_state_list, dtype=np.float32)
                            ).to(device)
                        )
                        # 2- get action value with selected next_action index
                        next_q_value = next_q_values[
                            list(experience.next_action.values())[0]
                        ]

                    elif self.algorithm == "Q-learning":
                        # greedy action as target behaviour for Q-learning
                        _, next_q_value, _ = self._act(
                            "greedy",
                            experience.next_state,
                            q_net=q_net,
                            device=device,
                        )

                    # update the target action values
                    if self.algorithm in ["Sarsa", "Q-learning"]:
                        # use the action value of the next action (Sarsa, Q-learning)
                        # Set reward tensor to avoid a Pytorch `RuntimeError`
                        target_q_value = torch.squeeze(
                            torch.Tensor([experience.reward]).to(device)
                            + discount_rate * next_q_value
                        )

                    batch_targets.append(target_q_value)

            # get the loss of the action value to update in the q net
            # detach indicates to not follow the gradient for the target, as it
            # implies the use of the q net too
            loss = loss_L1(
                torch.stack(batch_estimations), torch.stack(batch_targets).detach()
            )

            # Update the network
            optimizer.zero_grad()  # Reset the gradients an usual practice
            loss.backward()  # backpropagation Gradient descent
            optimizer.step()  # update network weights

            # update loss curve
            loss_curve.update(loss.item(), step, learning_rate=lr, epsilon=epsilon)

            # update reward curves
            if reward_curve_steps_per_point is not None:
                self._update_reward_curves(
                    environment,
                    q_net,
                    episode_start_state,
                    step,
                    device,
                    reward_curves,
                    reward_curve_mode,
                    reward_curve_steps_per_point,
                    lr,
                    epsilon,
                    training_best_reward,
                )

            # reduction of epsilon at each episode, with a min value of min_eps
            epsilon -= reduce_eps
            epsilon = max(epsilon, min_eps)

            # reduction of learning rate at each episode, with a min value of
            # min_lr
            lr -= lr * reduce_perc_lr / 100
            lr = max(lr, min_lr)
            # update optimizer with next learning rate [3]
            for g in optimizer.param_groups:
                g["lr"] = lr

            # update the step number
            step += 1

            # output convergence info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"The loss in step {step} has been {loss:.3}")

        # -------------------------------------------------------------------------
        # Step 2: save relevant training products
        # -------------------------------------------------------------------------
        self.q_net = q_net
        self.visited_states = visited_states
        self.known_states_list = known_states_list

        self.loss_curve = loss_curve
        if reward_curve_steps_per_point is not None:
            for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                if reward_curve_mode_ == "return":
                    self.reward_curve_return = reward_curves[idx]
                elif reward_curve_mode_ == "last_reward":
                    self.reward_curve_last_reward = reward_curves[idx]
                elif reward_curve_mode_ == "best_reward":
                    self.reward_curve_best_reward = reward_curves[idx]

        if save_q_net:  # [1]
            torch.save(
                q_net.state_dict(),
                f"./data/{self.save_folder}/q_net_{in_dim}_inputs.pt",
            )

        # -------------------------------------------------------------------------
        # Step 3: plot relevant data and save their figures and objects
        # -------------------------------------------------------------------------
        if plot_learning_curves:
            self._plot_learning_curves(
                environment.n_layers,
                loss_curve,
                reward_curves,
                reward_curve_mode,
                reward_reduction_factor=environment.reward_reduction_factor,
            )

    def load_net(
        self,
        in_dim: int,
        device: Literal["cuda", "mps", "cpu"],
    ):
        """Load agent network."""
        # Input dim is the number of aspects that define our state.
        # Output dim will be given by the number of possible actions for each
        # state, i. e., number of possible actions.
        out_dim = len(self.actions)

        # create the neural network object
        q_net = QNN(
            in_dim,
            out_dim,
            hidden_layers=self.hidden_layers,
            hidden_neur=self.hidden_neur,
        )

        # load weights and biases
        w_and_b = torch.load(f"./data/{self.save_folder}/q_net_{in_dim}_inputs.pt")
        # load weights and biases into created neural network object
        # employ `w_and_b` for the net before any operation to avoid consuming
        # the iterable
        q_net.load_state_dict(w_and_b)
        # load network into selected device to ensure we are employing always
        # the same device
        self.q_net = q_net.to(device)

        # output info about network number of layers and neurons inside them
        inputs_nd_n_neurons = [
            (weights.shape[1], weights.shape[0])
            for key, weights in w_and_b.items()
            if "weight" in key
        ]
        logging.info(
            """Loaded neural network with the following architecture (input
            layer not included):"""
        )
        for n_layer, shapes in enumerate(inputs_nd_n_neurons):
            logging.info(
                f"\tLayer {n_layer} with {shapes[0]} inputs and {shapes[1]} neurons"
            )

    def _update_reward_curves(
        self,
        environment: gym.Env,
        q_net: QNN,
        episode_start_state: str,
        step: int,
        device: Literal["cuda", "mps", "cpu"],
        reward_curves: list[learning_curve],
        reward_curve_mode: list[str],
        steps_per_point: int,
        lr: float | None = None,
        epsilon: float | None = None,
        training_best_reward: float | None = None,
    ):
        """Update selected reward curves in `reward_curve_mode` for each step.

        For that, we perform a greedy simulation along selected `steps_per_point`
        with the current state of the `q_net`.

        Parameters
        ----------
        environment: environment
            Environment of the simulation, deep copied to perform the greedy
            simulation.
        q_net: QNN
            Q-network with which make the simulation.
        episode_start_state: str
            Start state of an episode. Used as starting point of the simulation.
        step: int
            Step of the training, x-axis of reward curves.
        device: Literal["cuda", "mps", "cpu"]
            Currently used device for training. Used as an `act` method input.
        reward_curves: list[learning_curve]
            Rewards curves to update.
        reward_curve_mode: list[str]
            Reward curve modes to retrieve in order to update input reward
            curves.
        steps_per_point: int
            Number of steps of the simulation.
        lr: Optional[float], optional
            Learning rate value. If None, do not represent it in reward curves.
            By default, None.
        epsilon: Optional[float], optional
            Epsilon value. If None, do not represent it in reward curves. By
            default, None.
        training_best_reward: Optional[float], optional
            Training best reward for input step. Necessary in order to update
            best_reward learning curve.

        Warnings
        --------
        * It is assumed `reward_curve_mode` matches input `reward_curves` to
          update. TODO In future versions these inputs will be unified in order
          to assure it and secure the correct name/object assignation.

        References
        ----------
        .. [1] https://docs.python.org/3/library/copy.html
        """
        if len(reward_curves) != len(reward_curve_mode):
            raise warnings.warn(
                f"""Number of input reward curves does not match the number of
                reward curve modes. Only {reward_curve_mode} will be
                updated.""",
                stacklevel=1,
            )

        if "best_reward" in reward_curve_mode and training_best_reward is None:
            raise ValueError(
                """`training_best_reward` is required as input in order to
                update best_reward learning curve."""
            )

        # make a small simulation to get the rewards of the net for an episode
        overall_return, last_reward, _, _, _ = self.greedy_simulation(
            q_net=q_net,
            environment=copy.deepcopy(environment),  # [1]
            start_state=episode_start_state,
            steps=steps_per_point,
            device=device,
        )
        for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
            if reward_curve_mode_ == "return":
                step_reward = overall_return
            elif reward_curve_mode_ == "last_reward":
                step_reward = last_reward
            elif reward_curve_mode_ == "best_reward":
                step_reward = training_best_reward
            # update learning curves
            reward_curves[idx].update(step_reward, step, lr, epsilon)

    def _plot_learning_curves(
        self,
        n_layers: int,
        loss_curve: learning_curve,
        reward_curves: list[learning_curve],
        reward_curve_mode: list[str],
        reward_reduction_factor: str | None = None,
    ):
        """Plot learning curves adapted to `DRL_agent` outputs.

        Warnings
        --------
        * It is assumed `reward_curve_mode` matches input `reward_curves` to
          update. TODO In future versions these inputs will be unified in order
          to assure it and secure the correct name/object assignation.
        """
        if len(reward_curves) != len(reward_curve_mode):
            raise warnings.warn(
                "Number of input reward curves does not match the number of "
                f"reward curve modes. Only {reward_curve_mode} will be updated.",
                stacklevel=1,
            )
        # obtain units label
        units_label = HTC_units_label(reward_reduction_factor)

        loss_curve.plot(
            title="",
            xlabel="Training step",
            ylabel="MAE loss",
            plot_epsilon=True,
            plot_lr=False,
            save_path=f"./img/{self.save_folder}/Loss_learning_curve_{n_layers}.png",
        )

        if len(reward_curves) != 0:
            for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                if reward_curve_mode_ == "return":
                    reward_ylabel = f"Return ({units_label})"
                    suffix = "return"
                elif reward_curve_mode_ == "last_reward":
                    reward_ylabel = f"Last reward of greedy simulation ({units_label})"
                    suffix = "last_reward"
                elif reward_curve_mode_ == "best_reward":
                    reward_ylabel = f"Maximum reward ({units_label})"
                    suffix = "best_reward"

                reward_curves[idx].plot(
                    title="",
                    xlabel="Training step",
                    ylabel=reward_ylabel,
                    plot_epsilon=True,
                    plot_lr=False,
                    y_divisor=None,
                    save_path=f"./img/{self.save_folder}/Reward_learning_curve_{n_layers}_{suffix}.png",
                )


class DQN_agent(DRL_agent):
    """Deep Q-Network agent."""

    def __init__(
        self,
        algorithm: Literal["Q-learning", "double_Q-learning"],
        actions: np.typing.ArrayLike,
        save_folder: str = "DRL_results",
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Deep Q-Networks algorithms.

        Implemented modification with respect plain Deep Q-learning in
        `DRL_agent`:
         - Memory replay (always active)
         - Target network
         - Double estimation

        Notes
        -----
        `Target network` and `Double estimation` can be either selected or
        deactivated through `target_estimation_mode` at `train` method.
        By default `Double estimation` is selected.
        """
        # check for some errors
        if algorithm not in ["Q-learning", "double_Q-learning"]:
            raise ValueError(
                "Deep Q-learning agent can only be used for Q-learning algorithms."
                f"Input algorithm was {algorithm}."
            )

        # inherit from parent class
        super().__init__(
            algorithm,
            actions,
            save_folder,
            seed,
            verbose,
            **kwargs,
        )

    def train(
        self,
        device: Literal["cuda", "mps", "cpu"],
        environment: gym.Env,
        discount_rate: float = 0.99,
        lr: float = 0.1,
        epsilon: float = 1.0,
        batch_size: int = 64,
        max_steps: int = np.inf,
        tol_loss: float = 0.0,
        plot_learning_curves: bool = True,
        save_q_net: bool = True,
        memory_size: int = 10000,
        n_batch_per_step: int = 4,
        n_new_experiences_per_step: int = 1,
        target_estimation_mode: Literal[
            "regular", "target network", "double"
        ] = "double",
        n_steps_for_target_net_update: int = 1000,
        **kwargs,
    ):
        """Train a NN to predict action-state values from input states.

        Parameters
        ----------
        memory_size : int, optional
            Maximum length of memory replay.
            By default, 10000 [2]
        n_batch_per_step : int, optional
            Number of batches to retrieve from memory for each training step.
            By default, 4 [2]
        n_new_experiences_per_step: int, optional
            Number of new experiences to store each training step.
            By default, 1. [1]
        target_estimation_mode : Literal, optional
            Select if we train Deep Q-Networks (DQN) with the estimation of the
            Q value forming the train target with the:
                * "regular" : Network that is being trained (training network).
                * "target network" : Target network (adds the use of such network).
                * "double" : The combination of the training network and target
                  network (technique known as double DQN).
            By default, "double".
        n_steps_for_target_net_update : int, optional
            Number of steps to wait to update target network weights with those
            of the training Q-network. Only necessary if
            `target_estimation_mode` is "target network" or "double".
            By default, 1000. [1]

        See Also
        --------
        DRL_agent : parent class.
            Most of input parameters are described at its `.train` method
            dosctring.

        Notes
        -----
        * Currently, one pair action-state is being updated for each individual
          sample of the batch.
        * Notice that `max_steps` is not related to the number of steps per
          episode. For this, we have environment step count. In addition, the
          number of samples to update the network at each step is determined by
          batch parameter.
          Meanwhile, tabular method is trained for a number of episodes composed
          by its number of steps.
        * Notice that exist two `best_htc` trackings: `memory_best_htc` and
          `training_best_htc`. Learning curve `max_htc_vs_known_state` is only
          referred to the first one, `memory_best_htc`.

        References
        ----------
        .. [1] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        .. [2] Fundations of Deep Reinforcement Learning, Laura Graesser and Wah
            Loon Keng
        .. [3] https://docs.python.org/3/library/copy.html
        .. [4] https://iamholumeedey007.medium.com/copy-deepcopy-vs-clone-in-pytorch-e5b951b0cea3
        """
        # fix seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # info
        logging.info(f"Training the agent with {self.algorithm} algorithm...")

        # ------ KWARGS ------
        reduce_eps = kwargs.get("reduce_eps", 1e-3)
        min_eps = kwargs.get("min_eps", 0.0)

        reduce_perc_lr = kwargs.get("reduce_perc_lr", 1e-4)
        min_lr = kwargs.get("min_lr", 0.0)

        reward_curve_mode = kwargs.get(
            "reward_curve_mode", ["return", "last_reward", "best_reward"]
        )
        reward_curve_steps_per_point = kwargs.get("reward_curve_steps_per_point", 30)
        episode_start_state = kwargs.get("episode_start_state", "initial")

        decorrelated = kwargs.get("decorrelated", False)
        step_count = kwargs.get("step_count", False)

        debug_counter = kwargs.get("debug_counter", 1)

        # define episode start with reset method
        episode_start_state = environment.reset(episode_start_state)

        # initialize batch state
        initial_batch_state = episode_start_state

        # basic checks of input values
        if epsilon > 1 or epsilon < 0:
            raise ValueError("Epsilon must be a number between 0 and 1.")
        if lr > 1 or lr < 0:
            raise ValueError("Learnig rate must be a number between 0 and 1.")
        if discount_rate > 1 or discount_rate < 0:
            raise ValueError("Discount rate must be a number between 0 and 1.")
        if plot_learning_curves and reward_curve_steps_per_point is None:
            warnings.warn(
                """In order to plot reward curves,
                `reward_curve_steps_per_point` must be other than None.""",
                stacklevel=1,
            )
        if not set(reward_curve_mode) <= {"return", "last_reward", "best_reward"}:
            raise ValueError(
                """Invalid reward curve mode. Please, select 'return',
                'last_reward' or 'best_reward'."""
            )
        if target_estimation_mode not in ["regular", "target network", "double"]:
            raise ValueError(
                """Invalid `target estimation mode`. Please, select 'regular',
                'target network' or 'double'."""
            )

        # -------------------------------------------------------------------------
        # Step 0: define the NN of the Q function
        # -------------------------------------------------------------------------
        # Input dim is the number of aspects that define our state.
        # Take into account that, in our implementation, the state will be given
        # by a str with a collection of numbers.
        in_dim = len(str_to_tuple_or_list(environment.start_state, to="list"))

        # Output dim will be given by the number of possible actions for each
        # state, i. e., number of possible actions.
        out_dim = len(self.actions)

        # create the neural network
        q_net = QNN(
            in_dim,
            out_dim,
            hidden_layers=self.hidden_layers,
            hidden_neur=self.hidden_neur,
        )
        # and select its training hardware
        q_net = q_net.to(device)

        # set the optimizer
        optimizer = optim.Adam(q_net.parameters(), lr=lr)

        # set the loss
        loss_L1 = nn.L1Loss()

        if target_estimation_mode in ["target network", "double"]:
            # create the target neural network [4]
            q_net_target = copy.deepcopy(q_net)
            # initialize target network frequency counter
            target_net_counter = 0
        # -------------------------------------------------------------------------
        # Step 1: perform the training of the neural network
        # -------------------------------------------------------------------------
        # ------ INITIALIZATION ------
        # initialize step number, amount of loss, memory of experiences,
        # visited_states storage, storage of best reward seen along all the
        # training, storage of known states and storage of the higher htc
        # obtained in memory experiences
        step = 1
        loss = np.inf
        replay_memory = ReplayMemory(
            memory_size,
            self,
            n_experiences=batch_size,  # agent._experience_generation kwargs
            q_net=q_net,
            environment=environment,
            device=device,
            epsilon=epsilon,
            initial_state=initial_batch_state,
            follow_next_action=False,
            decorrelated=decorrelated,
            step_count=step_count,
        )
        visited_states = set()
        training_best_reward = -np.inf
        known_states_list = []

        # initialize learning curves
        loss_curve = learning_curve()
        reward_curves = []
        if reward_curve_steps_per_point is not None:
            for _ in reward_curve_mode:
                reward_curves.append(learning_curve())

            warnings.warn(
                """In order to save `reward_curves`, an additional simulation at
                each step of the algorithm is made. Please, consider if this
                type of plot is necessary. If not, set
                `reward_curve_steps_per_point` to None.""",
                stacklevel=1,
            )

        # ------ ALGORITHM ------
        # loop during a determined number of steps or until convergence
        while step <= max_steps and tol_loss < loss:
            # output step info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"Computing step number {step}...")

            # ----------------------------------------------------
            # Step 1.0: store new experiences in memory replay
            # ----------------------------------------------------
            # generate selected number of new experiences
            # continue experience storage from the state s' of the last experience
            new_experiences, _, _, _, _ = self._experience_generation(
                n_experiences=n_new_experiences_per_step,
                q_net=q_net,
                environment=environment,
                device=device,
                epsilon=epsilon,
                initial_state=replay_memory.memory[-1].next_state,
                follow_next_action=False,
                decorrelated=decorrelated,
                step_count=step_count,
            )

            # store the transition
            replay_memory.push(new_experiences)

            for _ in range(n_batch_per_step):
                # ----------------------------------------------------------------------
                # Step 1.1: retrieve desired number of experiences and store them
                # as a batch [1]
                # ----------------------------------------------------------------------
                batch = replay_memory.sample(batch_size, self.random_rng)

                # ----------------------------------------------------------------------
                # Step 1.2: use the batch experiences to get several pairs
                # estimation and target q values
                # ----------------------------------------------------------------------
                batch_estimations = []
                batch_targets = []
                for experience in batch:
                    # store unique visited states during the network training
                    # with the usage of set
                    visited_states.add(experience.state)

                    # Transform state form label to list[int]
                    state_list = str_to_tuple_or_list(experience.state, to="list")

                    # obtain ALL the q value estimations of the net for state
                    # It is necessary to set input as float32 so Pythorch does not
                    # return us a `RuntimeError` due dtypes
                    # Additionally, execute the forward pass at the same device we
                    # are using for training to avoid a Pytorch `RuntimeError`

                    estimated_q_values = q_net.forward(
                        torch.from_numpy(np.array(state_list, dtype=np.float32)).to(
                            device
                        )
                    )
                    batch_estimations.append(estimated_q_values[experience.action_idx])

                    # do not follow the gradient for obtained target values
                    with torch.no_grad():
                        if target_estimation_mode in ["regular", "double"]:
                            network_for_target_estimation = q_net

                        elif target_estimation_mode == "target network":
                            network_for_target_estimation = q_net_target

                        # greedy action as target behaviour for Q-learning
                        next_action_idx, next_q_value, _ = self._act(
                            "greedy",
                            experience.next_state,
                            q_net=network_for_target_estimation,
                            device=device,
                        )

                        # double DQN: take the q value from target network with
                        # action obtained from trained network
                        if target_estimation_mode == "double":
                            # next state
                            next_state_list = str_to_tuple_or_list(
                                experience.next_state, to="list"
                            )
                            # all `q_net_target` q values
                            target_net_q_values = q_net_target.forward(
                                torch.from_numpy(
                                    np.array(next_state_list, dtype=np.float32)
                                ).to(device)
                            )
                            # select next q value form target network with
                            # greedy action from trained network
                            next_q_value = target_net_q_values[next_action_idx]

                        # update the target action values
                        # use the action value of the next action (Sarsa, Q-learning)
                        # Set reward tensor to avoid a Pytorch `RuntimeError`
                        target_q_value = torch.squeeze(
                            torch.Tensor([experience.reward]).to(device)
                            + discount_rate * next_q_value
                        )

                        batch_targets.append(target_q_value)

                # store the best reward found during TRAINING
                if experience.reward > training_best_reward:
                    training_best_reward = experience.reward
                    logging.debug(
                        f"""Better training reward at step {step} and visited
                            state number {len(visited_states)}:
                            {experience.reward}"""
                    )
                # get the loss of the action value to update in the q net
                # detach indicates to not follow the gradient for the target, as it
                # implies the use of the q net too
                loss = loss_L1(
                    torch.stack(batch_estimations), torch.stack(batch_targets).detach()
                )

                # Update the network
                optimizer.zero_grad()  # Reset the gradients as usual practice
                loss.backward()  # backpropagation Gradient descent
                optimizer.step()  # update network weights

            # update loss curve when all batches per step have been processed
            loss_curve.update(loss.item(), step, learning_rate=lr, epsilon=epsilon)

            # update target network weights if `n_steps_for_target_net_update`
            # is reached
            if target_estimation_mode in ["target network", "double"]:
                target_net_counter += 1
                if target_net_counter == n_steps_for_target_net_update:
                    # copy q_net weight into target q_net
                    q_net_target.copy_weights(q_net)
                    # reset the counter
                    target_net_counter = 0

            # make a small simulation to get the rewards of the net for an
            # episode
            if reward_curve_steps_per_point is not None:
                overall_return, last_reward, _, _, _ = self.greedy_simulation(
                    q_net=q_net,
                    environment=copy.deepcopy(environment),
                    start_state=episode_start_state,
                    steps=reward_curve_steps_per_point,
                    device=device,
                )
                for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                    if reward_curve_mode_ == "return":
                        step_reward = overall_return
                    elif reward_curve_mode_ == "last_reward":
                        step_reward = last_reward
                    elif reward_curve_mode_ == "best_reward":
                        step_reward = training_best_reward

                    # update learning curves
                    reward_curves[idx].update(
                        step_reward, step, learning_rate=lr, epsilon=epsilon
                    )

            # reduction of epsilon at each episode, with a min value of min_eps
            epsilon -= reduce_eps
            epsilon = max(epsilon, min_eps)

            # reduction of learning rate at each episode, with a min value of
            # min_lr
            lr -= lr * reduce_perc_lr / 100
            lr = max(lr, min_lr)
            # update optimizer with next learning rate
            for g in optimizer.param_groups:
                g["lr"] = lr

            # update the step number
            step += 1

            # output convergence info if requested
            if self.verbose and step % debug_counter == 0:
                logging.debug(f"The loss in step {step} has been {loss:.3}")

        # -------------------------------------------------------------------------
        # Step 2: save relevant training products
        # -------------------------------------------------------------------------
        self.q_net = q_net
        self.visited_states = visited_states
        self.known_states_list = known_states_list

        self.loss_curve = loss_curve
        if reward_curve_steps_per_point is not None:
            for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                if reward_curve_mode_ == "return":
                    self.reward_curve_return = reward_curves[idx]
                elif reward_curve_mode_ == "last_reward":
                    self.reward_curve_last_reward = reward_curves[idx]
                elif reward_curve_mode_ == "best_reward":
                    self.reward_curve_best_reward = reward_curves[idx]

        if save_q_net:  # [1]
            torch.save(
                q_net.state_dict(),
                f"./data/{self.save_folder}/q_net_{in_dim}_inputs.pt",
            )

        # -------------------------------------------------------------------------
        # Step 3: plot relevant data and save their figures and objects
        # -------------------------------------------------------------------------
        if plot_learning_curves:
            # obtain units label
            units_label = HTC_units_label(environment.reward_reduction_factor)

            loss_curve.plot(
                title="",
                xlabel="Training step",
                ylabel="MAE loss",
                plot_epsilon=True,
                plot_lr=False,
                save_path=f"./img/{self.save_folder}/Loss_learning_curve_{environment.n_layers}.png",
            )

            if reward_curve_steps_per_point is not None:
                for idx, reward_curve_mode_ in enumerate(reward_curve_mode):
                    if reward_curve_mode_ == "return":
                        reward_ylabel = f"Return ({units_label})"
                        suffix = "return"
                    elif reward_curve_mode_ == "last_reward":
                        reward_ylabel = (
                            f"Last reward of greedy simulation ({units_label})"
                        )
                        suffix = "last_reward"
                    elif reward_curve_mode_ == "best_reward":
                        reward_ylabel = f"Maximum reward ({units_label})"
                        suffix = "best_reward"

                    reward_curves[idx].plot(
                        title="",
                        xlabel="Training step",
                        ylabel=reward_ylabel,
                        plot_epsilon=True,
                        plot_lr=False,
                        y_divisor=None,
                        save_path=f"./img/{self.save_folder}/Reward_learning_curve_{environment.n_layers}_{suffix}.png",
                    )
