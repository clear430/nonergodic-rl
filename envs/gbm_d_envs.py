#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  gbm_d_envs.py
python version:         3.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    OpenAI Gym compatible environments for training an agent on various gambles 
    for assets following geometric Brownian motion with discrete portfolio compounding
    based on 
    https://www.tandfonline.com/doi/pdf/10.1080/14697688.2010.513338?needAccess=true,
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.100603, and
    https://arxiv.org/pdf/1802.02939.pdf.
"""

import sys
sys.path.append("./")

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import List, Tuple

from extras.utils import multi_dones

MAX_VALUE = 1e18                                        # maximium potfolio value for normalisation
INITIAL_PRICE = 1e3                                     # intial price of all assets
INITIAL_VALUE = 1e4                                     # intial portfolio value
MIN_VALUE_RATIO = 1e-2                                  # minimum portfolio value ratio (psi)
MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
MAX_VALUE_RATIO = 1                                     # maximum possible value realtive to MAX_VALUE

MAX_ABS_ACTION = 0.99                                   # maximum normalised (absolute) action value (epsilon_1)
MIN_REWARD = 1e-6                                       # minimum step reward (epsilon_2)
MIN_RETURN = -0.99                                      # minimum step return (epsilon_3)
MIN_WEIGHT = 1e-6                                       # minimum all asset weights (epsilon_4)

# hyperparameters for investors 1-3 GBM gamble
DRIFT = 0.0516042410820218          # annual S&P500 mean log return over 120 years ending December 31
VOL = 0.190381677824107             # annual S&P500 sample standard deviation over 120 years ending December 31
LOG_MEAN = DRIFT - VOL**2 / 2       # mean of lognormal distribution for S&P500 prices

# maximum (absolute) leverage per assset (eta)
LEV_FACTOR = 6

class GBM_D_InvA(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the GBM approximation of the S&P500 index gamble.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_gambles: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_gambles: number of simultaneous identical gambles
        """
        super(GBM_D_InvA, self).__init__()

        self.n_gambles = n_gambles

        if n_gambles == 1:
            self.risk = np.empty((3 + n_gambles), dtype=np.float64)
        else:
            self.risk = np.empty((4 + n_gambles), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + n_gambles,), dtype=np.float64)

        # action space: [leverages]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(n_gambles,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flags for episode termination and whether genuine
            risk: collection of additional data retrieved from each step
        """
        initial_wealth = self.wealth
        assets = self.assets
        
        # obtain actions from neural network
        lev = action
        
        # sample new price % change factor
        dp = np.exp(np.random.normal(loc=LOG_MEAN, scale=VOL, size=self.n_gambles))

        # one-step portfolio return
        r = dp - 1
        step_return = np.sum(lev * r)

        # obtain next state
        self.wealth = initial_wealth * (1 + step_return)
        self.asset = assets * (1 + r)

        next_state = np.empty((1 + self.n_gambles), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # episode termination criteria
        done = multi_dones(self.wealth, MIN_VALUE, reward, MIN_REWARD, step_return, MIN_RETURN, 
                           lev, MIN_WEIGHT, next_state, MAX_VALUE_RATIO)

        self.risk[0:4] = [reward, self.wealth, step_return, np.mean(lev)]

        if self.n_gambles > 1:
            self.risk[4:] = lev

        self.time += 1

        return next_state, reward, done, self.risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = INITIAL_PRICE * np.ones((self.n_gambles), dtype=np.float64)

        state = np.empty((1 + self.n_gambles), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state

class GBM_D_InvB(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss]
    at each time step for the GBM approximation of the S&P500 index gamble.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_gambles: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_gambles: number of simultaneous identical gambles
        """
        super(GBM_D_InvB, self).__init__()

        self.n_gambles = n_gambles

        if n_gambles == 1:
            self.risk = np.empty((4 + n_gambles), dtype=np.float64)
        else:
            self.risk = np.empty((5 + n_gambles), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + n_gambles,), dtype=np.float64)

        # action space: [leverages]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(1 + n_gambles,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flags for episode termination and whether genuine
            risk: collection of additional data retrieved from each step
        """
        initial_wealth = self.wealth
        assets = self.assets
        
        # obtain actions from neural network
        stop_loss = (action[0] + 1) / 2
        lev = action[1:] * LEV_FACTOR
        
        # sample new price % change factor
        dp = np.exp(np.random.normal(loc=LOG_MEAN, scale=VOL, size=self.n_gambles))

        # one-step portfolio return
        r = dp - 1
        step_return = np.sum(lev * r)
        
        # amount of portoflio to bet and outcome
        min_wealth = INITIAL_VALUE * stop_loss
        active = initial_wealth - min_wealth
        change = active * (1 + step_return)

        # obtain next state
        self.wealth = min_wealth + change
        self.asset = assets * (1 + r)

        next_state = np.empty((1 + self.n_gambles), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # episode termination criteria
        done = multi_dones(self.wealth, min_wealth, reward, MIN_REWARD, step_return, MIN_RETURN, 
                           lev, MIN_WEIGHT, next_state, MAX_VALUE_RATIO)

        self.risk[0:5] = [reward, self.wealth, step_return, np.mean(lev), stop_loss]

        if self.n_gambles > 1:
            self.risk[5:] = lev

        self.time += 1

        return next_state, reward, done, self.risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = INITIAL_PRICE * np.ones((self.n_gambles), dtype=np.float64)

        state = np.empty((1 + self.n_gambles), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state

class GBM_D_InvC(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for the GBM approximation of the S&P500 index gamble.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_gambles: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_gambles: number of simultaneous identical gambles
        """
        super(GBM_D_InvC, self).__init__()

        self.n_gambles = n_gambles

        if n_gambles == 1:
            self.risk = np.empty((5 + n_gambles), dtype=np.float64)
        else:
            self.risk = np.empty((6 + n_gambles), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + n_gambles,), dtype=np.float64)

        # action space: [leverages]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2 + n_gambles,), dtype=np.float64)

        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            done: Boolean flags for episode termination and whether genuine
            risk: collection of additional data retrieved from each step
        """
        initial_wealth = self.wealth
        assets = self.assets
        
        # obtain actions from neural network
        stop_loss = (action[0] + 1) / 2
        retention = (action[1] + 1) / 2
        lev = action[2:] * LEV_FACTOR
        
        # sample new price % change factor
        dp = np.exp(np.random.normal(loc=LOG_MEAN, scale=VOL, size=self.n_gambles))

        # one-step portfolio return
        r = dp - 1
        step_return = np.sum(lev * r)
        
        # amount of portoflio to bet and outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to investor B risk-taking
            min_wealth = INITIAL_VALUE * stop_loss
            active = initial_wealth - min_wealth
        else:
            # bet portion of existing profit at each step
            min_wealth = INITIAL_VALUE + (initial_wealth - INITIAL_VALUE) * retention
            active = initial_wealth - min_wealth

        change = active * (1 + step_return)

        # obtain next state
        self.wealth = min_wealth + change
        self.asset = assets * (1 + r)

        next_state = np.empty((1 + self.n_gambles), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, min_wealth)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # episode termination criteria
        done = multi_dones(self.wealth, min_wealth, reward, MIN_REWARD, step_return, MIN_RETURN, 
                           lev, MIN_WEIGHT, next_state, MAX_VALUE_RATIO)

        self.risk[0:6] = [reward, self.wealth, step_return, np.mean(lev), stop_loss, retention]

        if self.n_gambles > 1:
            self.risk[6:] = lev

        self.time += 1

        return next_state, reward, done, self.risk

    def reset(self):
        """
        Reset the environment for a new agent episode.

        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = INITIAL_PRICE * np.ones((self.n_gambles), dtype=np.float64)

        state = np.empty((1 + self.n_gambles), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state