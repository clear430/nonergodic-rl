import gym
from gym import spaces
import numpy as np

INITIAL_VALUE = 100
UP_PROB = 0.5
UP_R = 0.5
DOWN_R = -0.4
ASYM_LIM = 0

BIGGER_PAYOFF = np.abs(DOWN_R) if np.abs(UP_R) > np.abs(DOWN_R) else -np.abs(UP_R) 
LEV_FACTOR = 1 / BIGGER_PAYOFF
LEV_FACTOR = LEV_FACTOR - ASYM_LIM if np.abs(UP_R) > np.abs(DOWN_R) else LEV_FACTOR + ASYM_LIM

MIN_STATE = 1e-5
MAX_ABS_ACTION = 0.999

class Investor1Env(gym.Env):
    def __init__(self):
        super(Investor1Env, self).__init__()

        self.reward_range = (-np.inf, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_STATE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [stop-loss, retention ratio, leverage]
        self.action_space = spaces.Box(low=np.array([-MAX_ABS_ACTION], dtype=np.float32), 
                                       high=np.array([MAX_ABS_ACTION], dtype=np.float32))

        self.reset()

    def step(self, action: np.ndarray):
        # rescale varaibles to 0 -> 1 domain
        lev = (action[0] + 1) / 2
        
        # sample binary return
        r = UP_R if np.random.binomial(n=1, p=UP_PROB, size=None) == 1 else DOWN_R
        
        initial_wealth = self.wealth

        self.wealth = initial_wealth * (1 + lev * r)

        reward = self.wealth - initial_wealth
        reward /= initial_wealth

        done = bool(self.wealth < 1e-3 or reward <= -1 or lev == 0)

        state = np.array([self.wealth], dtype=np.float32)

        return state, reward, done, {}

    def reset(self):
        self.wealth = INITIAL_VALUE
        state = np.array([self.wealth], dtype=np.float32)

        return state

class Investor2Env(gym.Env):
    def __init__(self):
        super(Investor2Env, self).__init__()

        self.reward_range = (-np.inf, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_STATE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage, stop_loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray):
        # rescale varaibles to 0 -> 1 domain
        lev = (action[0] + 1) / 2 * LEV_FACTOR
        stop_loss = (action[1] + 1) / 2
        
        # sample binary return
        r = UP_R if np.random.binomial(n=1, p=UP_PROB, size=None) == 1 else DOWN_R
        
        initial_wealth = self.wealth

        active = INITIAL_VALUE * (1 - stop_loss)
            
        change = active * (1 + lev * r)

        self.wealth = (initial_wealth - active) + change 

        reward = self.wealth - initial_wealth
        reward /= initial_wealth

        done = bool(self.wealth < INITIAL_VALUE * stop_loss or reward <= -1 or lev == 0)

        state = np.array([self.wealth], dtype=np.float32)
        
        return state, reward, done, {}

    def reset(self):
        self.wealth = INITIAL_VALUE
        state = np.array([self.wealth], dtype=np.float32)

        return state

class Investor3Env(gym.Env):
    def __init__(self):
        super(Investor3Env, self).__init__()

        self.reward_range = (-np.inf, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_STATE, high=np.inf, shape=(1,), dtype=np.float32)

        # action space: [leverage, stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float32)

        self.reset()

    def step(self, action: np.ndarray):
        # rescale varaibles to 0 -> 1 domain
        lev = (action[0] + 1) / 2 * LEV_FACTOR
        stop_loss = (action[1] + 1) / 2
        retention = (action[2] + 1) / 2
        
        # sample binary return
        r = UP_R if np.random.binomial(n=1, p=UP_PROB, size=None) == 1 else DOWN_R
        
        initial_wealth = self.wealth

        if initial_wealth <= INITIAL_VALUE:
            active = INITIAL_VALUE * (1 - stop_loss)
        else:
            active = (initial_wealth - INITIAL_VALUE) * (1 - retention)
            
        change = active * (1 + lev * r)

        self.wealth = (initial_wealth - active) + change 

        reward = self.wealth - initial_wealth
        reward /= initial_wealth

        done = bool(self.wealth < INITIAL_VALUE * stop_loss or reward <= -1 or lev == 0)

        state = np.array([self.wealth], dtype=np.float32)

        # print('${}, g {:1.8f}%, r {:1.0f}%, lev {:1.2f}%, stop {:1.2f}%, reten {:1.2f}%'.format(
        #        self.wealth, reward*100 , r*100, lev*100, stop_loss*100, retention*100))

        return state, reward, done, {}

    def reset(self):
        self.wealth = INITIAL_VALUE
        state = np.array([self.wealth], dtype=np.float32)

        return state