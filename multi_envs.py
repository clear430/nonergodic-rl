import gym
from gym import spaces
import numpy as np
from typing import Tuple

INITIAL_VALUE = 100         # intial portfolio value 
MIN_VALUE = 1e-2            # minimum portfolio value
MAX_ABS_ACTION = 0.999      # maximum normalised (absolute) action
MIN_RETURN = -0.999         # minimum step return for termination

# hyperparameters for Investors 1-3 gamble
UP_PROB = 0.5               # probability of up move
UP_R = 0.5                  # upside return (>=0)
DOWN_R = -0.4               # downside return (0<=)
if np.abs(UP_R) > np.abs(DOWN_R):
    LEV_FACTOR = 1 / np.abs(DOWN_R)
else:
    LEV_FACTOR = - 1 / np.abs(UP_R)

class Investor1_1x(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the equally likely +50%/-40% simple gamble.

    Methods:
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(1,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio return from state transition
            done: Boolean flag for episode termination
            risk: actual actions taken to manage portfolio risk
        """
        initial_wealth = self.wealth
        
        # rescale varaibles to 0 -> 1 domain
        lev = (action[0] + 1) / 2       # maximum |leverage| of unity
        
        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=None)==1, UP_R, DOWN_R)
        
        # obtain next state
        self.wealth = initial_wealth * (1 + lev * r)
        next_state = np.array([self.wealth], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE 
                    or reward <= -1 
                    or lev == 0)

        risk = np.array([r, lev], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        state = np.array([self.wealth], dtype=np.float32)

        return state

class Investor2_1x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop loss] at each 
    time step for the equally likely +50%/-40% simple gamble.

    Methods:
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage, stop_loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio return from state transition
            done: Boolean flag for episode termination
            risk: actual actions taken to manage portfolio risk
        """
        initial_wealth = self.wealth

        # rescale varaibles to 0 -> 1 domain
        lev = (action[0] + 1) / 2 * LEV_FACTOR    # maximum |leverage| to prevent wipe-out from one step
        stop_loss = (action[1] + 1) / 2
                
        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=None)==1, UP_R, DOWN_R)
        
        # amount of portfolio to be bet amd the outcome
        active = INITIAL_VALUE * (1 - stop_loss)
        change = active * (1 + lev * r)

        # obtain next state
        self.wealth = (initial_wealth - active) + change
        next_state = np.array([self.wealth], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < INITIAL_VALUE * stop_loss 
                    or reward <= MIN_RETURN 
                    or lev == 0)
        
        risk = np.array([r, lev, stop_loss], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        state = np.array([self.wealth], dtype=np.float32)

        return state

class Investor3_1x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop loss, retention ratio]
    at each time step for the equally likely +50%/-40% simple gamble.

    Methods:
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage, stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio return from state transition
            done: Boolean flag for episode termination
            risk: actual actions taken to manage portfolio risk
        """
        initial_wealth = self.wealth

        # rescale varaibles to 0 -> 1 domain
        lev = (action[0] + 1) / 2 * LEV_FACTOR    # maximum |leverage| to prevent wipe-out from one step
        stop_loss = (action[1] + 1) / 2
        retention = (action[2] + 1) / 2

        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=None)==1, UP_R, DOWN_R)
        
        # amount of portfolio to be bet amd the outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to Investor 2 risk-taking
            active = INITIAL_VALUE * (1 - stop_loss)
        else:
            # bet portion of existing profit at each step
            active = (initial_wealth - INITIAL_VALUE) * (1 - retention)
        
        change = active * (1 + lev * r)

        # obtain next state
        self.wealth = (initial_wealth - active) + change
        next_state = np.array([self.wealth], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < INITIAL_VALUE * stop_loss 
                    or reward <= MIN_RETURN 
                    or lev == 0)
        
        risk = np.array([r, lev, stop_loss, retention], dtype=np.float32)
        
        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        state = np.array([self.wealth], dtype=np.float32)

        return state

class Investor1_2x(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverages at each time 
    step for the equally likely +50%/-40% simple gamble for a portfoilio of 
    two identical assets.

    Methods:
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, asset 0, asset 1]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(3,), dtype=np.float32)

        # action space: [leverage 0, leverage 1]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio return from state transition
            done: Boolean flag for episode termination
            risk: actual actions taken to manage portfolio risk
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        
        # parameterise actions to be leverge portions
        total_lev = action[0]
        lev0_weight = (action[1] + 1) / 2

        lev0 = total_lev * lev0_weight
        lev1 = total_lev * (1 - lev0_weight)

        # sample binary returns
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        # obtain next state
        self.asset0 = initial_asset0 * (1 + r[0])
        self.asset1 = initial_asset1 * (1 + r[1])
        
        self.wealth = initial_wealth * (1 + lev0 * r[0] + lev1 * r[1])

        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE 
                    or reward <= -1 
                    or (lev0 == 0 and lev1 == 0))

        risk = np.array([r[0], r[1], lev0, lev1], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        return state
     
class Investor2_2x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverages, stop losses],
    at each time step for the equally likely +50%/-40% simple gamble for a 
    portfoilio of two identical assets.

    Methods:
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, asset 0, asset 1]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(3,), dtype=np.float32)

        # action space: [leverage 0, leverage 1, stop_loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio return from state transition
            done: Boolean flag for episode termination
            risk: actual actions taken to manage portfolio risk
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        
        # parameterise actions to be leverge portions
        total_lev = action[0]
        lev0_weight = (action[1] + 1) / 2
        
        lev0 = total_lev * lev0_weight
        lev1 = total_lev * (1 - lev0_weight)
       
        stop_loss = (action[2] + 1) / 2

        # sample binary returns
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        # amount of portfolio to be bet amd the outcome
        active = INITIAL_VALUE * (1 - stop_loss)
        change = active * (1 + lev0 * r[0] + lev1 * r[1])

        # obtain next state
        self.asset0 = initial_asset0 * (1 + r[0])
        self.asset1 = initial_asset1 * (1 + r[1])
        self.wealth = (initial_wealth - active) + change

        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < INITIAL_VALUE * stop_loss 
                    or reward <= -1 
                    or (lev0 == 0 and lev1 == 0))

        risk = np.array([r[0], r[1], lev0, lev1, stop_loss], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        return state

class Investor3_2x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverages, stop losses, 
    retention ratio], at each time step for the equally likely +50%/-40% simple 
    gamble for a portfoilio of two identical assets.

    Methods:
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, asset 0, asset 1]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(3,), dtype=np.float32)

        # action space: [leverage 0, leverage 1, stop_loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(4,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio return from state transition
            done: Boolean flag for episode termination
            risk: actual actions taken to manage portfolio risk
        """
        initial_wealth = self.wealth
        initial_asset0, initial_asset1 = self.asset0, self.asset1
        
        # parameterise actions to be leverge portions
        total_lev = action[0]
        lev0_weight = (action[1] + 1) / 2
        
        lev0 = total_lev * lev0_weight
        lev1 = total_lev * (1 - lev0_weight)
       
        stop_loss = (action[2] + 1) / 2
        retention = (action[2] + 1) / 2

        # sample binary returns
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        # amount of portfolio to be bet amd the outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to Investor 2 risk-taking
            active = INITIAL_VALUE * (1 - stop_loss)
        else:
            # bet portion of existing profit at each step
            active = (initial_wealth - INITIAL_VALUE) * (1 - retention)

        change = active * (1 + lev0 * r[0] + lev1 * r[1])

        # obtain next state
        self.asset0 = initial_asset0 * (1 + r[0])
        self.asset1 = initial_asset1 * (1 + r[1])
        self.wealth = (initial_wealth - active) + change

        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < INITIAL_VALUE * stop_loss 
                    or reward <= -1 
                    or (lev0 == 0 and lev1 == 0))

        risk = np.array([r[0], r[1], lev0, lev1, stop_loss, retention], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        return state