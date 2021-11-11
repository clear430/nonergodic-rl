#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  dice_flip_sh_envs.py
python version:         3.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1
s
Description:
    OpenAI Gym compatible environments for training an agent on various three-state
    dice roll gambles with and without insurance safe haven based on 
    https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import List, Tuple

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

# hyperparameters for the dice roll gamble for investors 1-3
UP_PROB = 1 / 6                                 # probability of up move
DOWN_PROB = 1 / 6                               # probability of down move
MID_PROB = 1 - (UP_PROB + DOWN_PROB)
UP_R = 0.5                                      # upside return (UP_R>MID_R>=0)
DOWN_R = -0.5                                   # downside return (DOWN_R<=MID_R)
MID_R = 0.05                                    # mid return

# hyperparameters for the insurance safe haven
SH_UP_R = -1                                    # safe haven upside return (<=MID_R)
SH_DOWN_R = 5                                   # safe haven downside return (>0)
SH_MID_R = -1                                   # safe haven mid return (<0)

# maximum (absolute) leverage per assset type (eta)
if np.abs(UP_R) > np.abs(DOWN_R):
    LEV_FACTOR = 1 / np.abs(DOWN_R)
else:
    LEV_FACTOR = 1 / np.abs(UP_R)

SH_LEV_FACTOR = 1 / np.abs(SH_UP_R)

class Dice_SH_n1_U(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the dice roll gamble without safe haven.

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_U, self).__init__()

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
            seed: unique seed for NumPy.
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
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, np.nan, np.nan, np.nan], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Dice_SH_n1_I(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the dice roll gamble with safe haven.

    Replicates Chapter 3 - Side Bets from
    https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_I, self).__init__()

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
            seed: unique seed for NumPy.
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
        lev = (action[0] + 1) / 2
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R 
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = lev * r + (1 - lev) * r_sh

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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, np.nan, np.nan, 1-lev], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Dice_SH_n1_InvA_U(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the dice roll gamble without safe haven.

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_InvA_U, self).__init__()

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
            seed: unique seed for NumPy.
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
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, np.nan, np.nan, np.nan], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Dice_SH_n1_InvA_I(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step 
    for the dice roll gamble with safe haven.

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_InvA_I, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        #  state space: [cumulative reward, asset 0, safe haven]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(3,), dtype=np.float64)

        # action space: [leverage 0, safe haven leverage]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(2,), dtype=np.float64)

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
        initial_asset_sh = self.asset_sh
        
        # obtain leverage from neural network
        lev = action[0]
        lev_sh = (action[1] + 1) / 2 * SH_LEV_FACTOR
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R 
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = lev * r + lev_sh * r_sh
        
        # obtain next state
        self.asset0 = initial_asset0 * (1 + r)
        self.asset_sh = initial_asset_sh * (1 + r_sh)

        self.wealth = initial_wealth * (1 + step_return)
        
        next_state = np.array([self.wealth, self.asset0, self.asset_sh], dtype=np.float64)
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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, np.nan, np.nan, lev_sh], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE
        self.asset_sh = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0, self.asset_sh], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Dice_SH_n1_InvB_U(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at each 
    time step for the dice roll gamble without safe haven.

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_InvB_U, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0]
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
            seed: unique seed for NumPy.
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
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, stop_loss, np.nan, np.nan], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Dice_SH_n1_InvB_I(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at each 
    time step for the dice roll gamble with safe haven.

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_InvB_I, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0, safe haven]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(3,), dtype=np.float64)

        # action space: [leverage 0, safe have leverage, stop-loss]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(3,), dtype=np.float64)

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
        initial_asset_sh = self.asset_sh
        
        # obtain leverages and stop-loss from neural network
        stop_loss = (action[0] + 1) / 2
        lev = action[1] * LEV_FACTOR
        lev_sh = (action[2] + 1) / 2 * SH_LEV_FACTOR
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R 
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = lev * r + lev_sh * r_sh
        
        # amount of portoflio to bet and outcome
        min_wealth = INITIAL_VALUE * stop_loss
        active = initial_wealth - min_wealth
        change = active * (1 + step_return)

        # obtain next state
        self.asset0 = initial_asset0 * (1 + r)
        self.asset_sh = initial_asset_sh * (1 + r_sh)

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0, self.asset_sh], dtype=np.float64)
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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, stop_loss, np.nan, lev_sh], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE
        self.asset_sh = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0, self.asset_sh], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Dice_SH_n1_InvC_U(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for the dice roll gamble without safe havem.

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_InvC_U, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0]
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
            seed: unique seed for NumPy.
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
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

        # one-step portfolio return
        step_return = lev * r
        
        # amount of portoflio to bet and outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to reinvestor B risk-taking
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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, stop_loss, retention, np.nan], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0], dtype=np.float64)
        state /= MAX_VALUE

        return state

class Dice_SH_n1_InvC_I(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss, 
    retention ratio] at each time step for the dice roll gamble with safe haven.

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
        Intialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_n1_InvC_I, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0, safe haven]
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE_RATIO, 
                                            shape=(3,), dtype=np.float64)

        # action space: [leverage 0, safe haven leverage, stop-loss, retention ratio]
        self.action_space = spaces.Box(low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, 
                                       shape=(4,), dtype=np.float64)

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
        initial_asset_sh = self.asset_sh
        
        # obtain leverages, stop-loss, and retention ratio from neural network
        stop_loss = (action[0] + 1) / 2
        retention = (action[1] + 1) / 2
        lev = action[2] * LEV_FACTOR
        lev_sh = (action[3] + 1) / 2 * SH_LEV_FACTOR
        
        # sample returns
        r = np.random.choice(3, p=[UP_PROB, DOWN_PROB, MID_PROB], size=1)
        r = np.where(r==0, UP_R, r)
        r = np.where(r==1, DOWN_R, r)
        r = np.where(r==2, MID_R, r)[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R 
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = lev * r + lev_sh * r_sh
        
        # amount of portoflio to bet and outcome
        if initial_wealth <= INITIAL_VALUE:
            # revert to reinvestor B risk-taking
            min_wealth = INITIAL_VALUE * stop_loss
            active = initial_wealth - min_wealth
        else:
            # bet portion of existing profit at each step
            min_wealth = INITIAL_VALUE + (initial_wealth - INITIAL_VALUE) * retention
            active = initial_wealth - min_wealth

        change = active * (1 + step_return)

        # obtain next state
        self.asset0 = initial_asset0 * (1 + r)
        self.asset_sh = initial_asset_sh * (1 + r_sh)

        self.wealth = min_wealth + change
        
        next_state = np.array([self.wealth, self.asset0, self.asset_sh], dtype=np.float64)
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
                    or np.any(next_state > MAX_VALUE_RATIO))

        risk = np.array([reward, self.wealth, step_return, lev, stop_loss, retention, lev_sh], dtype=np.float64)

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
        self.asset0 = INITIAL_PRICE
        self.asset_sh = INITIAL_PRICE

        state = np.array([self.wealth, self.asset0, self.asset_sh], dtype=np.float64)
        state /= MAX_VALUE

        return state