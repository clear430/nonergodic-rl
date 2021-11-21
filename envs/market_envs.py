#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  market_envs.py
python version:         3.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    OpenAI Gym compatible environments for training an agent on 
    simulated real market environments for all three investor categories.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import List, Tuple

MAX_VALUE = 1e6                                         # maximium potfolio value for normalisation
INITIAL_VALUE = 1e4                                     # intial portfolio value
MIN_VALUE_RATIO = 1e-2                                  # minimum portfolio value ratio (psi)
MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
MAX_VALUE_RATIO = 1                                     # maximum possible value realtive to MAX_VALUE

MAX_ABS_ACTION = 0.99                                   # maximum normalised (absolute) action value (epsilon_1)
MIN_REWARD = 1e-6                                       # minimum step reward (epsilon_2)
MIN_RETURN = -0.99                                      # minimum step return (epsilon_3)
MIN_WEIGHT = 1e-6                                       # minimum all asset weights (epsilon_4)

# maximum (absolute) leverage per assset (eta)
LEV_FACTOR = 3

class Market_InvA_D1(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverages at each time 
    step for a simulated real market using the MDP assumption.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed (unused)
        """
        super(Market_InvA_D1, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length

        if n_assets == 1:
            self.risk = np.empty((3 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0-(n-1)]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + n_assets,), dtype=np.float64)

        # action space: [leverage 0-(n-1)]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(n_assets,), dtype=np.float64)

        self.seed()
        self.reset(None)

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, next_assets: np.ndarray) \
            -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            actual_done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain leverages from neural network
        lev = action * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = next_state / assets - 1
        step_return = np.sum(lev * r)

        self.wealth = initial_wealth * (1 + step_return)
        self.assets = next_assets

        next_state = np.empty((1 + self.n_assets), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.time == self.time_length 
                    or self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > MAX_VALUE_RATIO))
        
        actual_done = [done, done and not self.time == self.time_length]
        
        self.risk[0:4] = [reward, self.wealth, step_return, np.mean(lev)]
        
        if self.n_assets > 1:
            self.risk[4:] = lev

        self.time += 1

        return next_state, reward, actual_done, self.risk

    def reset(self, assets: np.ndarray):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((1 + self.n_assets), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state

class Market_InvB_D1(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at 
    each time step for a simulated real market using the MDP assumption.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed (unused)
        """
        super(Market_InvB_D1, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length

        if n_assets == 1:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0-(n-1)]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + n_assets,), dtype=np.float64)

        # action space: [leverage 0-(n-1), stop-loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(1 + n_assets,), dtype=np.float64)

        self.seed()
        self.reset(None)

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, next_assets: np.ndarray) \
            -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            actual_done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain leverages from neural network
        stop_loss = (action[0] + 1) / 2
        lev = action[1:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = next_state / assets - 1
        step_return = np.sum(lev * r)

        # amount of portoflio to bet and outcome        
        min_wealth = INITIAL_VALUE * stop_loss
        active = initial_wealth - min_wealth
        change = active * (1 + step_return)

        self.wealth = min_wealth + change
        self.assets = next_assets

        next_state = np.empty((1 + self.n_assets), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.time == self.time_length 
                    or self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > MAX_VALUE_RATIO))
        
        actual_done = [done, done and not self.time == self.time_length]

        self.risk[0:5] = [reward, self.wealth, step_return, np.mean(lev), stop_loss]
        
        if self.n_assets > 1:
            self.risk[5:] = lev

        self.time += 1

        return next_state, reward, actual_done, self.risk

    def reset(self, assets: np.ndarray):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((1 + self.n_assets), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state

class Market_InvC_D1(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for a simulated real market using 
    the MDP assumption.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed (unused)
        """
        super(Market_InvC_D1, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length

        if n_assets == 1:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((6 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0-(n-1)]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + n_assets,), dtype=np.float64)

        # action space: [leverage 0-(n-1), stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2 + n_assets,), dtype=np.float64)

        self.seed()
        self.reset(None)

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, next_assets: np.ndarray) \
            -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            actual_done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain leverages from neural network
        stop_loss = (action[0] + 1) / 2
        retention = (action[1] + 1) / 2
        lev = action[2:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = next_state / assets - 1
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

        self.wealth = min_wealth + change
        self.assets = next_assets

        next_state = np.empty((1 + self.n_assets), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.time == self.time_length 
                    or self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > MAX_VALUE_RATIO))
        
        actual_done = [done, done and not self.time == self.time_length]

        self.risk[0:6] = [reward, self.wealth, step_return, np.mean(lev), stop_loss, retention]
        
        if self.n_assets > 1:
            self.risk[6:] = lev

        self.time += 1

        return next_state, reward, actual_done, self.risk

    def reset(self, assets: np.ndarray):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((1 + self.n_assets), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state

class Market_InvA_Dx(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverages at each time 
    step for a simulated real market using a non-MDP assumption incorporating
    multiple past states.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed (unused)
        """
        super(Market_InvA_Dx, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length - obs_days + 1
        self.obs_days = obs_days

        if n_assets == 1:
            self.risk = np.empty((3 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0-(n-1)]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + obs_days * n_assets,), dtype=np.float64)

        # action space: [leverage 0-(n-1)]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(n_assets,), dtype=np.float64)

        self.seed()
        self.reset(None)

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, next_assets: np.ndarray) \
            -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            actual_done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain leverages from neural network
        lev = action * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = (next_state / assets - 1)[0:self.n_assets]
        step_return = np.sum(lev * r)

        self.wealth = initial_wealth * (1 + step_return)
        self.assets = next_assets

        next_state = np.empty((1 + self.obs_days * self.n_assets), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.time == self.time_length 
                    or self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > MAX_VALUE_RATIO))
        
        actual_done = [done, done and not self.time == self.time_length]
        
        self.risk[0:4] = [reward, self.wealth, step_return, np.mean(lev)]
        
        if self.n_assets > 1:
            self.risk[4:] = lev

        self.time += 1

        return next_state, reward, actual_done, self.risk

    def reset(self, assets: np.ndarray):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((1 + self.obs_days * self.n_assets), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state

class Market_InvB_Dx(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at 
    each time step for a simulated real market using a non-MDP assumption 
    incorporating multiple past states.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed
        """
        super(Market_InvB_Dx, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length - obs_days + 1
        self.obs_days = obs_days

        if n_assets == 1:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0-(n-1)]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + obs_days * n_assets,), dtype=np.float64)

        # action space: [leverage 0-(n-1), stop-loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(1 + n_assets,), dtype=np.float64)

        self.seed()
        self.reset(None)

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, next_assets: np.ndarray) \
            -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            actual_done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain leverages from neural network
        stop_loss = (action[0] + 1) / 2
        lev = action[1:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = (next_state / assets - 1)[0:self.n_assets]
        step_return = np.sum(lev * r)

        # amount of portoflio to bet and outcome        
        min_wealth = INITIAL_VALUE * stop_loss
        active = initial_wealth - min_wealth
        change = active * (1 + step_return)

        self.wealth = min_wealth + change
        self.assets = next_assets

        next_state = np.empty((1 + self.obs_days * self.n_assets), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.time == self.time_length 
                    or self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > MAX_VALUE_RATIO))
        
        actual_done = [done, done and not self.time == self.time_length]

        self.risk[0:5] = [reward, self.wealth, step_return, np.mean(lev), stop_loss]
        
        if self.n_assets > 1:
            self.risk[5:] = lev

        self.time += 1

        return next_state, reward, actual_done, self.risk

    def reset(self, assets: np.ndarray):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((1 + self.obs_days * self.n_assets), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state

class Market_InvC_Dx(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for a simulated real market using a 
    non-MDP assumption incorporating multiple past states.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int):
        """
        Intialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed
        """
        super(Market_InvC_Dx, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length - obs_days + 1
        self.obs_days = obs_days

        if n_assets == 1:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((6 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0-(n-1)]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(1 + obs_days * n_assets,), dtype=np.float64)

        # action space: [leverage 0-(n-1), stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2 + n_assets,), dtype=np.float64)

        self.seed()
        self.reset(None)

    def seed(self, seed=None) -> List[int]:
        """
        Fix randomisation seed.

        Parameters:
            seed: unique seed for NumPy.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, next_assets: np.ndarray) \
            -> Tuple[np.ndarray, float, List[bool], np.ndarray]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean
            actual_done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain leverages from neural network
        stop_loss = (action[0] + 1) / 2
        retention = (action[1] + 1) / 2
        lev = action[2:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = (next_state / assets - 1)[0:self.n_assets]
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

        self.wealth = min_wealth + change
        self.assets = next_assets

        next_state = np.empty((1 + self.obs_days * self.n_assets), dtype=np.float64)
        next_state[0], next_state[1:] = self.wealth, self.assets
        next_state /= MAX_VALUE

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.maximum(self.wealth, MIN_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)
        
        # episode termination criteria
        done = bool(self.time == self.time_length 
                    or self.wealth == MIN_VALUE
                    or reward < MIN_REWARD
                    or step_return < MIN_RETURN
                    or np.all(np.abs(lev) < MIN_WEIGHT)
                    or np.any(next_state > MAX_VALUE_RATIO))
        
        actual_done = [done, done and not self.time == self.time_length]

        self.risk[0:6] = [reward, self.wealth, step_return, np.mean(lev), stop_loss, retention]
        
        if self.n_assets > 1:
            self.risk[6:] = lev

        self.time += 1

        return next_state, reward, actual_done, self.risk

    def reset(self, assets: np.ndarray):
        """
        Reset the environment for a new agent episode.
        
        Parameters:
            state: default intial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((1 + self.obs_days * self.n_assets), dtype=np.float64)
        state[0], state[1:] = self.wealth, self.assets

        state /= MAX_VALUE

        return state