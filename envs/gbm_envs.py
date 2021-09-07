import gym
from gym import spaces
from gym.utils import seeding
from matplotlib import scale
import numpy as np
from typing import Tuple

INITIAL_VALUE = 100         # intial portfolio value 
MIN_VALUE = 1e0             # minimum portfolio value (important for Q-value convergence)
MAX_ABS_ACTION = 0.99       # maximum normalised (absolute) action value
MIN_RETURN = -0.99          # minimum return required for episode termination
MAX_ABS_LEV = 10            # maximum total absolute leverage

# hyperparameters for investors 1-3 GBM gamble
RISK_FREE = 0.005            # risk-free rate of return
RETURN_LOC = 0               # mean excess return of any asset
RETURN_SCALE = 0.01          # std about mean excess return of any asset
VOL_LOC = 0.1                # mean std of any asset 
VOL_SCALE = 0.1              # std about mean std of any asset

class Investor1GBM_1x(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for a asset following GBM.

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
        super(Investor1GBM_1x, self).__init__()

        self.reward_range = (-1, np.inf)

        #  state space: [cumulative reward]
        self.observation_space = spaces.Box(low=MIN_VALUE, high=np.inf, 
                                            shape=(1,), dtype=np.float32)

        # action space: [leverage]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(1,), dtype=np.float32)

        self.seed()

        self.asset_returns = np.random.normal(loc=RETURN_LOC, scale=RETURN_SCALE, size=None)
        self.asset_vols = np.abs(np.random.normal(loc=VOL_LOC, scale=VOL_SCALE, size=None))

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
        
        # leverage with maximum scaling
        lev = action[0] * MAX_ABS_LEV

        # geometric brownian motion
        mu  = RISK_FREE + lev * self.asset_returns - (lev * self.asset_vols)**2 / 2
        wiener = np.random.standard_normal(size=None)
        var = lev * self.asset_vols * wiener
        r = mu + var
        
        # obtain next state
        self.wealth = initial_wealth * np.exp(r)
        next_state = np.array([self.wealth], dtype=np.float32)

        # calculate the step reward as a return
        reward = (self.wealth - initial_wealth) / initial_wealth

        # episode termination criteria
        done = bool(self.wealth < MIN_VALUE 
                    or lev == 0
                    or reward <= MIN_RETURN)

        optimal_lev = self.asset_returns / self.asset_vols**2

        risk = np.array([self.asset_returns, self.asset_vols, optimal_lev, lev], dtype=np.float32)

        return next_state, reward, done, risk

    def reset(self):
        """
        Reset the environment for a new agent episode.
        """
        self.wealth = INITIAL_VALUE

        state = np.array([self.wealth], dtype=np.float32)

        return state