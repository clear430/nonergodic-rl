import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import Tuple

INITIAL_VALUE = 100         # intial portfolio value 
MIN_VALUE = 1e0             # minimum portfolio value (important for Q-value convergence)
MAX_ABS_ACTION = 0.99       # maximum normalised (absolute) action value
MIN_RETURN = -0.99          # minimum return required for episode termination 

# hyperparameters for investors 1-3 gamble
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(1,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        lev = (action[0] + 1) / 2 * np.sign(LEV_FACTOR)    # maximum |leverage| of unity
        
        # sample binary return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=None)==1, UP_R, DOWN_R)
        
        # obtain next state
        self.wealth = initial_wealth * (1 + lev * r)
        next_state = np.array([self.wealth], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE 
                    or lev == 0
                    or reward <= MIN_RETURN)

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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage, stop_loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
                    or lev == 0
                    or reward <= MIN_RETURN)
        
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage, stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
                    or lev == 0
                    or reward <= MIN_RETURN)
        
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, asset 0, asset 1]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(3,), dtype=np.float32)

        # action space: [total leverage, psuedo weight 0]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        lev0_psuedo_weight = (action[1] + 1) / 2

        # construct actual leverage for each asset
        lev0 = total_lev * lev0_psuedo_weight
        lev1 = total_lev * (1 - lev0_psuedo_weight)

        # sample binary returns and construct total return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        weighted_r = lev0 * r[0] + lev1 * r[1]

        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])

        self.wealth = initial_wealth * (1 + weighted_r)

        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE 
                    or (lev0 == 0 and lev1 == 0)
                    or reward <= MIN_RETURN)

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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, asset 0, asset 1]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(3,), dtype=np.float32)

        # action space: [total leverage, psuedo weight 0, stop loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        total_lev = action[0]  * LEV_FACTOR
        lev0_psuedo_weight = (action[1] + 1) / 2

        # construct actual leverage for each asset
        lev0 = total_lev * lev0_psuedo_weight
        lev1 = total_lev * (1 - lev0_psuedo_weight)
       
        stop_loss = (action[2] + 1) / 2

        # sample binary returns and construct total return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        weighted_r = lev0 * r[0] + lev1 * r[1]

        # amount of portfolio to be bet amd the outcome
        active = INITIAL_VALUE * (1 - stop_loss)
        change = active * (1 + weighted_r)

        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])

        self.wealth = (initial_wealth - active) + change

        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < INITIAL_VALUE * stop_loss
                    or (lev0 == 0 and lev1 == 0)
                    or reward <= MIN_RETURN)                    

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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, asset 0, asset 1]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(3,), dtype=np.float32)

        # action space: [leverage 0, leverage 1, stop_loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(4,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        total_lev = action[0] * LEV_FACTOR
        lev0_psuedo_weight = (action[1] + 1) / 2 
        
        # construct actual leverage for each asset
        lev0 = total_lev * lev0_psuedo_weight
        lev1 = total_lev * (1 - lev0_psuedo_weight)
       
        stop_loss = (action[2] + 1) / 2
        retention = (action[3] + 1) / 2

        # sample binary returns and construct total return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=2)==1, UP_R, DOWN_R)

        weighted_r = lev0 * r[0] + lev1 * r[1]

        # amount of portfolio to be bet amd the outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to Investor 2 risk-taking
            active = INITIAL_VALUE * (1 - stop_loss)
        else:
            # bet portion of existing profit at each step
            active = (initial_wealth - INITIAL_VALUE) * (1 - retention)

        change = active * (1 + weighted_r)

        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])

        self.wealth = (initial_wealth - active) + change

        next_state = np.array([self.wealth, self.asset0, self.asset1], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < INITIAL_VALUE * stop_loss 
                    or (lev0 == 0 and lev1 == 0)
                    or reward <= MIN_RETURN)

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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, assets 0-9]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(11,), dtype=np.float32)

        # action space: [total leverage, psudeo weights 0-9]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(11,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        initial_asset2, initial_asset3 = self.asset2, self.asset3
        initial_asset4, initial_asset5 = self.asset4, self.asset5
        initial_asset6, initial_asset7 = self.asset6, self.asset7
        initial_asset8, initial_asset9 = self.asset8, self.asset9
        
        # parameterise actions to be (normalised) weighted leverge portions
        total_lev = action[0]
        psuedo_weight = action[1:]

        norm = np.sum(psuedo_weight)
        psuedo_weight /= norm

        lev = psuedo_weight * total_lev

        # sample binary returns and construct total return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=10)==1, UP_R, DOWN_R)

        weighted_r = np.sum(lev * r)

        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])
        self.asset2, self.asset3 = initial_asset2 * (1 + r[2]), initial_asset3 * (1 + r[3])
        self.asset4, self.asset5 = initial_asset4 * (1 + r[4]), initial_asset5 * (1 + r[5])
        self.asset6, self.asset7 = initial_asset6 * (1 + r[6]), initial_asset7 * (1 + r[7])
        self.asset8, self.asset9 = initial_asset8 * (1 + r[8]), initial_asset9 * (1 + r[9])

        self.wealth = initial_wealth * (1 + weighted_r)

        next_state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                               self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                               self.asset9], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE
                    or np.all(lev == 0)
                    or reward <= MIN_RETURN)

        risk = np.array([total_lev], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        self.asset2, self.asset3 = INITIAL_VALUE, INITIAL_VALUE
        self.asset4, self.asset5 = INITIAL_VALUE, INITIAL_VALUE
        self.asset6, self.asset7 = INITIAL_VALUE, INITIAL_VALUE
        self.asset8, self.asset9 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                          self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                          self.asset9], dtype=np.float32)

        return state

class Investor2_10x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverages, stop losses] 
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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, assets 0-9]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(11,), dtype=np.float32)

        # action space: [total leverage, psudeo weights 0-9, stop loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(12,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        initial_asset2, initial_asset3 = self.asset2, self.asset3
        initial_asset4, initial_asset5 = self.asset4, self.asset5
        initial_asset6, initial_asset7 = self.asset6, self.asset7
        initial_asset8, initial_asset9 = self.asset8, self.asset9
        
        # parameterise actions to be (normalised) weighted leverge portions
        total_lev = action[0]
        psuedo_weight = action[1:11]

        norm = np.sum(psuedo_weight)
        psuedo_weight /= norm

        lev = psuedo_weight * total_lev

        stop_loss = (action[11] + 1) / 2

        # sample binary returns and construct total return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=10)==1, UP_R, DOWN_R)

        weighted_r = np.sum(lev * r)

        # amount of portfolio to be bet amd the outcome
        active = INITIAL_VALUE * (1 - stop_loss)
        change = active * (1 + weighted_r)

        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])
        self.asset2, self.asset3 = initial_asset2 * (1 + r[2]), initial_asset3 * (1 + r[3])
        self.asset4, self.asset5 = initial_asset4 * (1 + r[4]), initial_asset5 * (1 + r[5])
        self.asset6, self.asset7 = initial_asset6 * (1 + r[6]), initial_asset7 * (1 + r[7])
        self.asset8, self.asset9 = initial_asset8 * (1 + r[8]), initial_asset9 * (1 + r[9])

        self.wealth = (initial_wealth - active) + change

        next_state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                               self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                               self.asset9], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE
                    or np.all(lev == 0)
                    or reward <= MIN_RETURN)

        risk = np.array([total_lev, stop_loss], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        self.asset2, self.asset3 = INITIAL_VALUE, INITIAL_VALUE
        self.asset4, self.asset5 = INITIAL_VALUE, INITIAL_VALUE
        self.asset6, self.asset7 = INITIAL_VALUE, INITIAL_VALUE
        self.asset8, self.asset9 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                          self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                          self.asset9], dtype=np.float32)

        return state

class Investor3_10x(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverages, stop losses, 
    retention ratios] at each time step for the equally likely +50%/-40% simple gamble 
    for a portfoilio of ten identical assets.

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

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward, assets 0-9]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(11,), dtype=np.float32)

        # action space: [total leverage, psudeo weights 0-9, stop loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(13,), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        initial_asset2, initial_asset3 = self.asset2, self.asset3
        initial_asset4, initial_asset5 = self.asset4, self.asset5
        initial_asset6, initial_asset7 = self.asset6, self.asset7
        initial_asset8, initial_asset9 = self.asset8, self.asset9
        
        # parameterise actions to be (normalised) weighted leverge portions
        total_lev = action[0]
        psuedo_weight = action[1:11]

        norm = np.sum(psuedo_weight)
        psuedo_weight /= norm

        lev = psuedo_weight * total_lev

        stop_loss = (action[11] + 1) / 2
        retention = (action[12] + 1) / 2

        # sample binary returns and construct total return
        r = np.where(np.random.binomial(n=1, p=UP_PROB, size=10)==1, UP_R, DOWN_R)

        weighted_r = np.sum(lev * r)

        # amount of portfolio to be bet amd the outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to Investor 2 risk-taking
            active = INITIAL_VALUE * (1 - stop_loss)
        else:
            # bet portion of existing profit at each step
            active = (initial_wealth - INITIAL_VALUE) * (1 - retention)

        change = active * (1 + weighted_r)

        # obtain next state
        self.asset0, self.asset1 = initial_asset0 * (1 + r[0]), initial_asset1 * (1 + r[1])
        self.asset2, self.asset3 = initial_asset2 * (1 + r[2]), initial_asset3 * (1 + r[3])
        self.asset4, self.asset5 = initial_asset4 * (1 + r[4]), initial_asset5 * (1 + r[5])
        self.asset6, self.asset7 = initial_asset6 * (1 + r[6]), initial_asset7 * (1 + r[7])
        self.asset8, self.asset9 = initial_asset8 * (1 + r[8]), initial_asset9 * (1 + r[9])

        self.wealth = (initial_wealth - active) + change

        next_state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                               self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                               self.asset9], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE
                    or np.all(lev == 0)
                    or reward <= MIN_RETURN)

        risk = np.array([total_lev, stop_loss, retention], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE
        self.asset0, self.asset1 = INITIAL_VALUE, INITIAL_VALUE
        self.asset2, self.asset3 = INITIAL_VALUE, INITIAL_VALUE
        self.asset4, self.asset5 = INITIAL_VALUE, INITIAL_VALUE
        self.asset6, self.asset7 = INITIAL_VALUE, INITIAL_VALUE
        self.asset8, self.asset9 = INITIAL_VALUE, INITIAL_VALUE

        state = np.array([self.wealth, self.asset0, self.asset1, self.asset2, self.asset3,
                          self.asset4, self.asset5, self.asset6, self.asset7, self.asset9, 
                          self.asset9], dtype=np.float32)

        return state