import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import List, Tuple

MIN_VALUE = 1e0             # minimum portfolio value (important for Q-value convergence)
INITIAL_VALUE = 1e2         # intial portfolio value 
MAX_VALUE = 1e16            # maximium potfolio value for normalisation
MAX_VALUE_RATIO = 1         # maximum possible value realtive to MAX_VALUE

MAX_ABS_ACTION = 0.99       # maximum normalised (absolute) action value (epsilon_1)
MIN_REWARD = 1e-6           # minimum step reward (epsilon_2)
MIN_RETURN = -0.99          # minimum step return (epsilon_3)
MIN_WEIGHT = 1e-6           # minimum all asset weights (epsilon_4)

# hyperparameters for the coin flip gamble for investors 1-3
UP_PROB = 0.5               # probability of up move
UP_R = 0.5                  # upside return (>=0)
DOWN_R = -0.4               # downside return (0<=)

# maximum (absolute) portfolio leverage (eta)
if np.abs(UP_R) > np.abs(DOWN_R):
    LEV_FACTOR = 1 / np.abs(DOWN_R)
else:
    LEV_FACTOR = 1 / np.abs(UP_R)

class Investor1_1x(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the equally likely +50%/-40% simple gamble.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor1_1x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(2,), dtype=np.float64)

        # action space: [leverage 0]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(1,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0 = self.asset0
        
        # obtain leverage from neural network
        lev = action[0]
        
        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=None)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = lev * r
        
        # obtain next state
        self.asset0 = initial_asset0 * (1 + r)

        self.wealth = initial_wealth * (1 + step_return)
        
        next_state = np.array([self.wealth, self.asset0], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # episode termination criteria
        done = bool(self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.abs(lev) < MIN_WEIGHT
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, lev], dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0 = INITIAL_VALUE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor1_2x(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the equally likely +50%/-40% simple gamble for a portfoilio of 
    two identical assets.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor1_2x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0-1]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(3,), dtype=np.float64)

        # action space: [leverage 0-1]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        
        # obtain leverages from neural network
        lev = action

        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = np.sum(lev * r)
        
        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])

        self.wealth = initial_wealth * (1 + step_return)
        
        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, lev[0], lev[1]], dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor1_10x(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverages at each time 
    step for the equally likely +50%/-40% simple gamble for a portfoilio of 
    ten identical assets.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor1_10x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0-1]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(11,), dtype=np.float64)

        # action space: [leverage 0-9]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(10,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        initial_asset2, initial_asset3 = self.asset2, self.asset3
        initial_asset4, initial_asset5 = self.asset4, self.asset5
        initial_asset6, initial_asset7 = self.asset6, self.asset7
        initial_asset8, initial_asset9 = self.asset8, self.asset9
        
        # obtain leverages from neural network
        lev = action

        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=10)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = np.sum(lev * r)
        
        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])
        self.asset2, self.asset3 = initial_asset2 * (1 + r[2]), initial_asset3 * (1 + r[3])
        self.asset4, self.asset5 = initial_asset4 * (1 + r[4]), initial_asset5 * (1 + r[5])
        self.asset6, self.asset7 = initial_asset6 * (1 + r[6]), initial_asset7 * (1 + r[7])
        self.asset8, self.asset9 = initial_asset8 * (1 + r[8]), initial_asset9 * (1 + r[9])

        self.wealth = initial_wealth * (1 + step_return)
        
        next_state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                               self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                               self.asset9], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, lev[0], lev[1], lev[2], lev[3], 
                         lev[4], lev[5], lev[6], lev[7], lev[8], lev[9]], dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        self.asset2, self.asset3 = INITIAL_VALUE, INITIAL_VALUE
        self.asset4, self.asset5 = INITIAL_VALUE, INITIAL_VALUE
        self.asset6, self.asset7 = INITIAL_VALUE, INITIAL_VALUE
        self.asset8, self.asset9 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                          self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                          self.asset9], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor2_1x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at each 
    time step for the equally likely +50%/-40% simple gamble.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor2_1x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(2,), dtype=np.float64)

        # action space: [leverage 0, stop-loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0 = self.asset0
        
        # obtain leverages and stop-loss from neural network
        stop_loss = (action[0] + 1) / 2
        lev = action[1] * LEV_FACTOR
        
        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=None)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = lev * r
        
        # amount of portoflio to bet and outcome
        min_wealth = INITIAL_VALUE * stop_loss
        active = initial_wealth - min_wealth
        change = active * (1 + step_return)

        # obtain next state
        self.asset0 = initial_asset0 * (1 + r)

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # episode termination criteria
        done = bool(self.wealth == min_wealth
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.abs(lev) < MIN_WEIGHT
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, lev, stop_loss], dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0 = INITIAL_VALUE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor2_2x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at each 
    time step for the equally likely +50%/-40% simple gamble for a portfoilio of 
    two identical assets.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor2_2x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0-1]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(3,), dtype=np.float64)

        # action space: [leverage 0-1, stop-loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        
        # obtain leverages and stop-loss from neural network
        stop_loss = (action[0] + 1) / 2
        lev = action[1:] * LEV_FACTOR

        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = np.sum(lev * r)

        # amount of portoflio to bet and outcome        
        min_wealth = INITIAL_VALUE * stop_loss
        active = initial_wealth - min_wealth
        change = active * (1 + step_return)
        
        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.wealth == min_wealth
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, stop_loss, lev[0], lev[1]], dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor2_10x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss]
    at each time step for the equally likely +50%/-40% simple gamble for a 
    portfoilio of ten identical assets.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor2_10x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0-1]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(11,), dtype=np.float64)

        # action space: [leverage 0-9, stop-loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(11,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        initial_asset2, initial_asset3 = self.asset2, self.asset3
        initial_asset4, initial_asset5 = self.asset4, self.asset5
        initial_asset6, initial_asset7 = self.asset6, self.asset7
        initial_asset8, initial_asset9 = self.asset8, self.asset9
        
        # obtain leverages and stop-loss from neural network
        stop_loss = (action[0] + 1) / 2
        lev = action[1:] * LEV_FACTOR

        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=10)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = np.sum(lev * r)

        # amount of portoflio to bet and outcome        
        min_wealth = INITIAL_VALUE * stop_loss
        active = initial_wealth - min_wealth
        change = active * (1 + step_return)
        
        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])
        self.asset2, self.asset3 = initial_asset2 * (1 + r[2]), initial_asset3 * (1 + r[3])
        self.asset4, self.asset5 = initial_asset4 * (1 + r[4]), initial_asset5 * (1 + r[5])
        self.asset6, self.asset7 = initial_asset6 * (1 + r[6]), initial_asset7 * (1 + r[7])
        self.asset8, self.asset9 = initial_asset8 * (1 + r[8]), initial_asset9 * (1 + r[9])

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                               self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                               self.asset9], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.wealth == min_wealth
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, stop_loss, lev[0], lev[1], lev[2], 
                         lev[3], lev[4], lev[5], lev[6], lev[7], lev[8], lev[9]], 
                         dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        self.asset2, self.asset3 = INITIAL_VALUE, INITIAL_VALUE
        self.asset4, self.asset5 = INITIAL_VALUE, INITIAL_VALUE
        self.asset6, self.asset7 = INITIAL_VALUE, INITIAL_VALUE
        self.asset8, self.asset9 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                          self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                          self.asset9], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor3_1x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for the equally likely +50%/-40% simple gamble.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor3_1x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(2,), dtype=np.float64)

        # action space: [leverage 0, stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0 = self.asset0
        
        # obtain leverages, stop-loss, and retention ratio from neural network
        stop_loss = (action[0] + 1) / 2
        retention = (action[1] + 1) / 2
        lev = action[2] * LEV_FACTOR
        
        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=None)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = lev * r
        
        # amount of portoflio to bet and outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to investor 2 risk-taking
            min_wealth = INITIAL_VALUE * stop_loss
            active = initial_wealth - min_wealth
        else:
            # bet portion of existing profit at each step
            min_wealth = INITIAL_VALUE + (initial_wealth - INITIAL_VALUE) * retention
            active = initial_wealth - min_wealth

        change = active * (1 + step_return)

        # obtain next state
        self.asset0 = initial_asset0 * (1 + r)

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # episode termination criteria
        done = bool(self.wealth == min_wealth
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.abs(lev) < MIN_WEIGHT
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, stop_loss, retention, lev], 
                        dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0 = INITIAL_VALUE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor3_2x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for the equally likely +50%/-40% 
    simple gamble for a portfoilio of two identical assets.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor3_2x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0-1]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(3,), dtype=np.float64)

        # action space: [leverage 0-1, stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(4,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        
        # obtain leverages, stop-loss, and retention ratio from neural network
        stop_loss = (action[0] + 1) / 2
        retention = (action[1] + 1) / 2
        lev = action[2:] * LEV_FACTOR

        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = np.sum(lev * r)

        # amount of portoflio to bet and outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to investor 2 risk-taking
            min_wealth = INITIAL_VALUE * stop_loss
            active = initial_wealth - min_wealth
        else:
            # bet portion of existing profit at each step
            min_wealth = INITIAL_VALUE + (initial_wealth - INITIAL_VALUE) * retention
            active = initial_wealth - min_wealth

        change = active * (1 + step_return)
        
        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.wealth == min_wealth
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, stop_loss, retention,
                         lev[0], lev[1]], dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Investor3_10x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for the equally likely +50%/-40% 
    simple gamble for a portfoilio of ten identical assets.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self):
        """
        Intialise class varaibles by creating state-action space and reward range
        """
        super(Investor3_10x, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0-1]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(11,), dtype=np.float64)

        # action space: [leverage 0-9, stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(12,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for Numpy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flag for episode termination
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        initial_asset2, initial_asset3 = self.asset2, self.asset3
        initial_asset4, initial_asset5 = self.asset4, self.asset5
        initial_asset6, initial_asset7 = self.asset6, self.asset7
        initial_asset8, initial_asset9 = self.asset8, self.asset9
        
        # obtain leverages, stop-loss, and retention ratio from neural network
        stop_loss = (action[0] + 1) / 2
        retention = (action[1] + 1) / 2
        lev = action[2:] * LEV_FACTOR

        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=10)==1, UP_R, DOWN_R)

        # one-step portfolio return
        step_return = np.sum(lev * r)

        # amount of portoflio to bet and outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to investor 2 risk-taking
            min_wealth = INITIAL_VALUE * stop_loss
            active = initial_wealth - min_wealth
        else:
            # bet portion of existing profit at each step
            min_wealth = INITIAL_VALUE + (initial_wealth - INITIAL_VALUE) * retention
            active = initial_wealth - min_wealth

        change = active * (1 + step_return)
        
        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])
        self.asset2, self.asset3 = initial_asset2 * (1 + r[2]), initial_asset3 * (1 + r[3])
        self.asset4, self.asset5 = initial_asset4 * (1 + r[4]), initial_asset5 * (1 + r[5])
        self.asset6, self.asset7 = initial_asset6 * (1 + r[6]), initial_asset7 * (1 + r[7])
        self.asset8, self.asset9 = initial_asset8 * (1 + r[8]), initial_asset9 * (1 + r[9])

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                               self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                               self.asset9], dtype=np.float64)
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.wealth == min_wealth
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > 1))

        risk = np.array([self.wealth, reward, step_return, stop_loss, retention, lev[0], 
                         lev[1], lev[2], lev[3], lev[4], lev[5], lev[6], lev[7], lev[8], 
                         lev[9]], dtype=np.float64)

        self.time += 1

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        self.asset2, self.asset3 = INITIAL_VALUE, INITIAL_VALUE
        self.asset4, self.asset5 = INITIAL_VALUE, INITIAL_VALUE
        self.asset6, self.asset7 = INITIAL_VALUE, INITIAL_VALUE
        self.asset8, self.asset9 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                          self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                          self.asset9], dtype=np.float64)
        state /= MAX_VALUE

        return state