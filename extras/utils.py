#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  utils.py
python version:         3.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for various additional tools required for file naming, directory 
    generation, shadow means, unique time series generation and aggregating 
    output training data for final figure plotting.
"""

import numpy as np
import scipy.optimize as op
import scipy.special as sp
from typing import List, Tuple

def save_directory(inputs: dict, results: bool) -> str:
    """
    Provides string directory for data and plot saving names.

    Parameters:
        inputs: dictionary containg all execution details
        results: whether results (True) or model (False)

    Returns:
        directory: file path and name to give to current experiment plots
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    if inputs['dynamics'] == 'A':
        dyna = 'additive/' 
    elif inputs['dynamics'] == 'M':
        dyna =  'multiplicative/'
    else:
        dyna = 'market/'    

    dir = ['./results/', 
           dyna,
           inputs['env_id']+'/',
           inputs['env_id']+'--',
           inputs['dynamics']+'_',
           inputs['algo']+'-',
           inputs['s_dist'],
           '_'+inputs['loss_fn'],
           '-'+str(inputs['critic_mean_type']),
           '_B'+str(int(inputs['buffer']))[0:2]+'e'+str(buff_exp-1),
           '_M'+str(inputs['multi_steps']),
           '_S'+str(int(inputs['n_cumsteps']))[0:2]+'e'+str(step_exp-1),
           '_N'+str(inputs['n_trials'])  
           ]

    if results == False:
        dir[0] = './models/'

    directory = ''.join(dir)

    return directory

def plot_subtitles(inputs: dict):
    """
    Generate subtitles for plots and figures.

    Parameters:
        inputs: dictionary containg all execution details
    
    Returns:
        sub: subtitle to be used in plots
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    sub = [inputs['env_id']+'--',
           inputs['dynamics']+'_',
           inputs['algo']+'-',
           inputs['s_dist'],
           '_'+inputs['loss_fn'],
           '-'+str(inputs['critic_mean_type']),
           '_B'+str(int(inputs['buffer']))[0:2]+'e'+str(buff_exp-1),
           '_M'+str(inputs['multi_steps']),
           '_S'+str(int(inputs['n_cumsteps']))[0:2]+'e'+str(step_exp-1),
           '_N'+str(inputs['n_trials'])  
           ]
    
    sub = ''.join(sub)
    
    return sub

def multi_log_dim(inputs: dict, n_gambles: int) -> int:
    """
    Generates risk-related parameter log dimension for multiplicative experiments 
    with dimensions dependent on the environment characteristics.

    Parameters
        inputs: dictionary containg all execution details
        n_gambles: number of simultaneous identical gambles
        
    Returns:
        dim: dimensions for log array
    """
    env = inputs['env_id']
    
    dim = 4
    
    if n_gambles > 1:
        dim += n_gambles    

    if '_InvB' in env:
        dim += 1
    if '_InvC' in env:
        dim += 2
               
    if '_SH' in env:
        dim = 4 + 2 + 1

    return dim

def market_log_dim(inputs: dict, n_assets: int) -> int:
    """
    Generates risk-related parameter log dimension for market experiments 
    with dimensions dependent on the environment characteristics.

    Parameters
        inputs: dictionary containg all execution details
        n_assets: number of assets for leverages
        
    Returns:
        dim: dimensions for log array
    """
    env = inputs['env_id']
    
    dim = 4
    
    if n_assets > 1:
        dim += n_assets    
    
    if '_InvB' in env:
        dim += 1
    if '_InvC' in env:
        dim += 2
        
    return dim

def get_exponent(array: np.ndarray) -> int:
    """
    Obtain expoenent for maximum array value used for scaling and axis labels.

    Parameters:
        array: array of usually cumulative steps in trial

    Returns:
        exp: exponent of max cumulative steps
    """
    max_step = np.max(array)

    if str(max_step)[0] == 1:
        exp = int(len(str(int(max_step))))
    else:
        exp = int(len(str(int(max_step))) - 1)

    return exp

def multi_dones(wealth: float, MIN_VALUE: float, reward: float, MIN_REWARD: float, 
                step_return: float, MIN_RETURN: float, lev: np.ndarray, MIN_WEIGHT: float,
                next_state: np.ndarray, MAX_VALUE_RATIO: float) \
        -> List[bool]:
    """
    Agent done flags for multiplicative environments controlling episode termination and 
    Q-value estimation for learning (i.e. whether genuine).

    Parameters:
        wealth: portfolio value
        reward: time-average growth rate
        step_return: single step return
        lev: asset leverages
        next_state: normalised state values

    Returns:
        done: episode termination
        learn_done: agent learning flag
    """
    done_wealth = wealth == MIN_VALUE
    done_reward = reward < MIN_REWARD
    done_return = step_return < MIN_RETURN
    done_lev = np.all(np.abs(lev)) < MIN_WEIGHT
    done_state = np.any(next_state > MAX_VALUE_RATIO)

    done = bool(done_wealth or done_reward or done_return or
                done_lev or done_state)

    learn_done = done and not done_state

    return [done, learn_done]

def market_dones(time: int, TIME_LENGTH: int, wealth: float, MIN_VALUE: float, 
                 reward: float, MIN_REWARD: float, step_return: float, MIN_RETURN: float, 
                 lev: np.ndarray, MIN_WEIGHT: float, next_state: np.ndarray, MAX_VALUE_RATIO: float) \
        -> List[bool]:
    """
    Agent done flags for multiplicative environments controlling episode termination and 
    Q-value estimation for learning (i.e. whether genuine).

    Parameters:
        time: episode time step
        wealth: portfolio value
        reward: time-average growth rate
        step_return: single step return
        lev: asset leverages
        next_state: normalised state values

    Returns:
        done: episode termination
        learn_done: agent learning flag
    """
    done_time = time == TIME_LENGTH
    done_wealth = wealth == MIN_VALUE
    done_reward = reward < MIN_REWARD
    done_return = step_return < MIN_RETURN
    done_lev = np.all(np.abs(lev)) < MIN_WEIGHT
    done_state = np.any(next_state > MAX_VALUE_RATIO)

    done = bool(done_time or done_wealth or done_reward or 
                done_return or done_lev or done_state)

    learn_done = done and not (done_time or done_state)

    return [done, learn_done]

def shadow_means(alpha: np.ndarray, min: np.ndarray, max: np.ndarray, 
                 min_mul: float, max_mul: float) \
        -> np.ndarray:
    """
    Construct shadow mean given the tail exponent and sample min/max for varying multipliers.

    Parameters:
        alpha: sample tail index
        min: sample minimum critic loss
        max: sample maximum critic loss
        low_mul: lower bound multiplier of sample minimum to form threshold of interest
        max_mul: upper bound multiplier of sample maximum to form upper limit

    Returns:
        shadow: shadow mean
    """
    low, high = min * min_mul, max * max_mul
    up_gamma = sp.gamma(1 - alpha) * sp.gammaincc(1 - alpha, alpha / high)
    shadow = low + (high - low) * np.exp(alpha / high) * (alpha / high)**alpha * up_gamma

    return shadow

def shadow_equiv(mean: np.ndarray, alpha: np.ndarray, min: np.ndarray, 
                 max: np.ndarray, min_mul: float =1) \
        -> np.ndarray:
    """
    Estimate max multiplier required for equivalence between empirical (arthmetic) mean
    and shadow mean estimate.

    Parameters:
        mean: empirical mean
        alpha: sample tail index
        min: sample minimum critic loss
        max: sample maximum critic loss
        low_mul: lower bound multiplier of sample minimum to form minimum threshold of interest

    Returns:
        max_mul:upper bound multiplier of maximum of distributions for equivalent
    """
    # select intial guess of equivilance multiplier
    x0 = 1

    if alpha < 1:
        f = lambda max_mul: shadow_means(alpha, min, max, min_mul, max_mul) - mean
        max_mul_solve = op.root(f, x0, method='hybr')
        max_mul_solve = max_mul_solve.x
    else:
        max_mul_solve = x0

    return max_mul_solve

def time_slice(prices: np.ndarray, extract_days: int, sample_days: int) -> Tuple[np.ndarray, int]:
    """
    Extract sequential slice of time series preserving the non-i.i.d. nature of the data
    keeping heteroscedasticity and serial correlation relatively unchanged compared to random sampling.

    Parameters:
        prices: array of all assets prices across a shared time period
        extract_days: length of period to be extracted
        sample_days: maximum buffer of days restricting starting index

    Returns:
        market_extract: extracted time sliced data from complete time series
        start_idx: first index of sampled training time series
    """
    max_train = prices.shape[0] - (1 + sample_days)

    start_idx = np.random.randint(0, max_train)
    end_idx = start_idx + extract_days + 1

    market_extract = prices[start_idx:end_idx]

    return market_extract, start_idx

def shuffle_data(prices: np.ndarray, interval_days: int) -> np.ndarray:
    """
    Split data into identical subset intervals and randomly shuffle data within each interval.
    Purpose is to generate a fairly random (non-parametric) bootstrap (or seed) for known 
    historical data while preserving overall long-term trends and structure.

    Parameters:
        prices: array of all historical assets prices across a shared time period
        interval_days: size of ordered subsets

    Returns:
        shuffled_prices: prices randomly shuffled within each interval
    """
    length = prices.shape[0]
    mod = length%interval_days
    intervals = int((length - mod) / interval_days) 

    shuffled_prices = np.empty((length, prices.shape[1]))
    
    if mod != 0:
        split_prices = np.split(prices[:-mod], indices_or_sections=intervals, axis=0)
    else:
        split_prices = np.split(prices, indices_or_sections=intervals, axis=0)

    for x in range(intervals):
        shuffled_prices[x * interval_days: (x + 1) * interval_days] = np.random.permutation(split_prices[x])

    if mod != 0:
        shuffled_prices[intervals * interval_days:] = np.random.permutation(prices[-mod:])

    return shuffled_prices

def train_test_split(prices: np.ndarray, train_years: float, test_years: float, gap_years: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequential train/test split of time series data with fixed gap between sets. This formulation 
    preserves the non-i.i.d. nature of the data keeping heteroscedasticity and serial correlation relatively 
    unchanged compared to random sampling.

    Parameters:
        prices: array of all assets prices across a shared time period
        train_years: length of training period
        test_years: length of testing period
        gap_years: fixed length between end of training and start of testing periods

    Returns:
        train: array of training period
        test: array of evaluation period
    """
    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    gap_days = int(252 * gap_years)

    max_train = prices.shape[0] - (1 + train_days + gap_days + test_days)

    start_train = np.random.randint(train_days - 1, max_train)
    end_train = start_train + train_days + 1
    start_test = end_train + gap_days
    end_test = start_test + test_days + 1

    train, test = prices[start_train:end_train], prices[start_test:end_test]

    return train, test

def add_loss_aggregate(env_keys: list, gym_envs: dict, inputs: dict, algos: list =['TD3'], loss: list =['MSE']) \
         -> np.ndarray:
    """
    Combine environment loss evaluation data for additive experiments.

    Parameters:
        env_keys: list of environments
        gym_envvs: dictionary of all environment details
        inputs: dictionary of additive execution parameters
        algos: list of RL algorithms
        loss: list of critic loss functions

    Retuens:
        data: aggregated evaluation data across all ioss functions
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    data = np.zeros((len(env_keys), len(algos), len(loss), int(inputs['n_trials']), 
                    int(inputs['n_cumsteps'] / inputs['eval_freq']), int(inputs['n_eval']), 20))

    name = [gym_envs[str(key)][0] for key in env_keys]
    path = ['./results/additive/' + n + '/' for n in name]

    num1 = 0
    for env in name:
        num2 = 0
        for a in algos:
            num3 = 0
            for l in loss:

                dir = ['--',
                    'A_',
                    a+'-',
                    inputs['s_dist'],
                    '_'+l,
                    '-'+str(inputs['critic_mean_type']),
                    '_B'+str(int(inputs['buffer']))[0:2]+'e'+str(buff_exp-1),
                    '_M'+str(inputs['multi_steps']),
                    '_S'+str(int(inputs['n_cumsteps']))[0:2]+'e'+str(step_exp-1),
                    '_N'+str(inputs['n_trials'])  
                    ]

                dir = ''.join(dir)

                data_path = path[num1]+env+dir

                file = np.load(data_path+'_eval.npy')

                data[num1, num2, num3] = file

                num3 += 1
            num2 += 1
        num1 += 1

    return data

def add_multi_aggregate(env_keys: list, gym_envs: dict, inputs: dict, algos: list =['TD3'], multi: list =[1]) \
         -> np.ndarray:
    """
    Combine environment multi-step evaluation data for additive experiments.

    Parameters:
        env_keys: list of environments
        gym_envvs: dictionary of all environment details
        inputs: dictionary of additive execution parameters
        algos: list of RL algorithms
        loss: list of multi-steps

    Retuens:
        data: aggregated evaluation data across all ioss functions
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    data = np.zeros((len(env_keys), len(algos), len(multi), int(inputs['n_trials']), 
                    int(inputs['n_cumsteps'] / inputs['eval_freq']), int(inputs['n_eval']), 20))

    name = [gym_envs[str(key)][0] for key in env_keys]
    path = ['./results/additive/' + n + '/' for n in name]

    num1 = 0
    for env in name:
        num2 = 0
        for a in algos:
            num3 = 0
            for m in multi:

                dir = ['--',
                    'A_',
                    a+'-',
                    inputs['s_dist'],
                    '_MSE',
                    '-'+str(inputs['critic_mean_type']),
                    '_B'+str(int(inputs['buffer']))[0:2]+'e'+str(buff_exp-1),
                    '_M'+str(m),
                    '_S'+str(int(inputs['n_cumsteps']))[0:2]+'e'+str(step_exp-1),
                    '_N'+str(inputs['n_trials'])  
                    ]

                dir = ''.join(dir)

                data_path = path[num1]+env+dir

                file = np.load(data_path+'_eval.npy')

                data[num1, num2, num3] = file

                num3 += 1
            num2 += 1
        num1 += 1

    return data

def add_summary(inputs: dict, data: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seperate and aggregate arrays into variables.
    
    Parameters:
        inputs: dictionary of execution parameters
        data: aggregated evaluation data across all experiments

    Retuens:
        reward: final scores
        loss: critic loss
        scale: Cauchy scales
        kernerl: CIM kernel size
        logtemp: SAC log entropy temperature
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means

    """
    n_env, n_algo, n_data = data.shape[0], data.shape[1], data.shape[2]

    count_x = int(inputs['n_cumsteps'] / inputs['eval_freq'])
    count_y = int(inputs['n_trials'] * int(inputs['n_eval']))
    count_z = int(inputs['n_trials'] )

    reward = np.zeros((n_env, n_algo, n_data, count_x, count_y))
    loss = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    scale = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    kernel = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    logtemp = np.zeros((n_env, n_algo, n_data, count_x, count_z))

    shadow = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    tail = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    cmin = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    cmax = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    keqv = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2)) 

    for e in range(n_env):
        for a in range(n_algo):
            for d in range(n_data):
                for t in range(count_x):
                    for n in range(inputs['n_trials']):

                        loss[e, a, d, t, (n * 2):(n * 2) + 2] = data[e, a, d, n, t, 0, 3:5]
                        scale[e, a, d, t, (n * 2):(n * 2) + 2] = data[e, a, d, n, t, 0, 15:17]
                        kernel[e, a, d, t, (n * 2):(n * 2) + 2] = data[e, a, d, n, t, 0, 17:19]
                        logtemp[e, a, d, t, n] = data[e, a, d, n, t, 0, 14]

                        shadow[e, a, d, t, (n * 2):(n * 2) + 2] = data[e, a, d, n, t, 0, 9:11]
                        tail[e, a, d, t, (n * 2):(n * 2) + 2] = data[e, a, d, n, t, 0, 11:13]
                        cmin[e, a, d, t, (n * 2):(n * 2) + 2] = data[e, a, d, n, t, 0, 5:7]
                        cmax[e, a, d, t, (n * 2):(n * 2) + 2] = data[e, a, d, n, t, 0, 7:9]

                        for s in range(int(inputs['n_eval'])):
                            reward[e, a, d, t, s + n * int(inputs['n_eval'])] = data[e, a, d, n, t, s, 1]

    shadow[np.isnan(shadow)] = loss[np.isnan(shadow)]

    for e in range(n_env):
        for a in range(n_algo):
            for d in range(n_data):
                for t in range(count_x):
                    for n in range(inputs['n_trials'] * 2):
                        keqv[e, a, d, t, n] = shadow_equiv(loss[e, a, d, t, n], tail[e, a, d, t, n], 
                                                           cmin[e, a, d, t, n], loss[e, a, d, t, n], 1)

    return reward, loss, scale, kernel, logtemp, tail, shadow, cmax, keqv

def mul_inv_aggregate(env_keys: list, n_gambles: int, gym_envs: dict, inputs: dict, safe_haven: bool =False) \
        -> np.ndarray:
    """
    Combine environment evaluation data for investors across the same number of assets.

    Parameters:
        env_keys: list of environments
        n_gambles: number of simultaneous identical gambles
        gym_envvs: dictionary of all environment details
        inputs: dictionary of multiplicative execution parameters
        safe_have: whether investor is using insurance safe haven

    Retuens:
        eval: aggregated evaluation data across all investors
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    dir = ['--',
           'M_',
           inputs['algo']+'-',
           inputs['s_dist'],
           '_'+inputs['loss_fn'],
           '-'+str(inputs['critic_mean_type']),
           '_B'+str(int(inputs['buffer']))[0:2]+'e'+str(buff_exp-1),
           '_M'+str(inputs['multi_steps']),
           '_S'+str(int(inputs['n_cumsteps']))[0:2]+'e'+str(step_exp-1),
           '_N'+str(inputs['n_trials'])  
           ]
    
    dir = ''.join(dir)
    
    sh = 1 if safe_haven == True else 0

    name = [gym_envs[str(key)][0] + '_n' + str(n_gambles) for key in env_keys]
    path = ['./results/multiplicative/' + n + '/' for n in name]

    eval = np.zeros((len(name), int(inputs['n_trials']), 
                        int(inputs['n_cumsteps'] / inputs['eval_freq']), 
                        int(inputs['n_eval']), 20 + 16 + sh))
    num = 0
    for env in name:
        data_path = path[num]+env+dir

        file1 = np.load(data_path+'_eval.npy')
        file2 = np.load(data_path+'_eval_risk.npy')
        file = np.concatenate((file1, file2), axis=3)

        eval[num, :, :, :, :20 + file2.shape[3]] = file

        num += 1

    return eval

def mul_inv_n_summary(mul_inputs: dict, aggregate_n: np.ndarray, safe_haven: bool =False) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seperate and aggregate arrays into variables.
    
    Parameters:
        inputs: dictionary of execution parameters
        aggregate_n: aggregated evaluation data across all investors

    Retuens:
        reward: 1 + time-average growth rate
        lev: leverages
        stop: stop-losses
        reten: retention ratios
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        lev_sh: leverage for safe haven
    """
    ninv = aggregate_n.shape[0]

    count_x = int(mul_inputs['n_cumsteps'] / mul_inputs['eval_freq'])
    count_y = int(mul_inputs['n_trials'] * int(mul_inputs['n_eval']))
    count_z = int(mul_inputs['n_trials'] )

    loss = np.zeros((ninv, count_x, count_z * 2))
    shadow = np.zeros((ninv, count_x, count_z * 2))
    tail = np.zeros((ninv, count_x, count_z * 2))
    cmin = np.zeros((ninv, count_x, count_z * 2))
    cmax = np.zeros((ninv, count_x, count_z * 2))
    keqv = np.zeros((ninv, count_x, count_z * 2)) 

    reward = np.zeros((ninv, count_x, count_y))
    lev = np.zeros((ninv, count_x, count_y))
    stop = np.zeros((ninv, count_x, count_y))
    reten = np.zeros((ninv, count_x, count_y))
    lev_sh = np.zeros((ninv, count_x, count_y))

    for i in range(ninv):
        for t in range(count_x):
            for n in range(mul_inputs['n_trials']):
                
                    loss[i, t, (n * 2):(n * 2) + 2] = aggregate_n[i, n, t, 0, 3:5]
                    shadow[i, t, (n * 2):(n * 2) + 2] = aggregate_n[i, n, t, 0, 9:11]
                    tail[i, t, (n * 2):(n * 2) + 2] = aggregate_n[i, n, t, 0, 11:13]
                    cmin[i, t, (n * 2):(n * 2) + 2] = aggregate_n[i, n, t, 0, 5:7]
                    cmax[i, t, (n * 2):(n * 2) + 2] = aggregate_n[i, n, t, 0, 7:9]

                    for s in range(int(mul_inputs['n_eval'])):
                        reward[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 20]
                        lev[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 23]
                        stop[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 24]
                        reten[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 25]

                        if safe_haven == True:
                            lev_sh[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 26]

    shadow[np.isnan(shadow)] = loss[np.isnan(shadow)]

    for i in range(ninv):
        for t in range(count_x):
            for n in range(mul_inputs['n_trials'] * 2):
                keqv[i, t, n] = shadow_equiv(loss[i, t, n], tail[i, t, n], 
                                             cmin[i, t, n], loss[i, t, n], 1)

    return reward, lev, stop, reten, loss, tail, shadow, cmax, keqv, lev_sh