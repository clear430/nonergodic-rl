#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  eval_episodes.py
python version:         3.9
torch verison:          1.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for performing agent evaluation episodes for both 
    additive and multiplicative environments.
"""

import sys
sys.path.append("./")

from datetime import datetime
import gym
import numpy as np
import pybullet_envs
import time
from typing import List, NoReturn, Tuple

import envs.coin_flip_envs as coin_flip_envs
import envs.dice_roll_envs as dice_roll_envs
import envs.dice_roll_sh_envs as dice_roll_sh_envs
import envs.market_envs  as market_envs
import envs.gbm_envs as gbm_envs
import envs.gbm_d_envs as gbm_d_envs
import extras.utils as utils

def eval_additive(agent: object, inputs: dict, eval_log: np.ndarray, cum_steps: int, round: int, 
                  eval_run: int, loss: List[np.ndarray], logtemp: np.ndarray, loss_params: List[np.ndarray]) \
        -> NoReturn:
    """
    Evaluates agent policy on environment without learning for a fixed number of episodes.

    Parameters:
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of existing evalaution results
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    print('{} {}-{}-{}-{} {} Evaluations at cst {}: C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}'
            .format(datetime.now().strftime('%d %H:%M:%S'), 
                    inputs['algo'], inputs['s_dist'], inputs['loss_fn'], round+1, int(inputs['n_eval']), cum_steps,
                    np.mean(loss[0:2]), np.mean(loss[4:6]), np.mean(loss[6:8]), 
                    np.mean(loss[8:10]), np.mean(loss_params[0:2]), np.mean(loss_params[2:4]), loss[8]+3, np.exp(logtemp)))
    
    eval_env = gym.make(inputs['env_id'])
    
    for eval in range(int(inputs['n_eval'])):
        start_time = time.perf_counter()
        run_state = eval_env.reset()
        run_done, run_step, run_reward = False, 0, 0

        while not run_done:
            run_action = agent.eval_next_action_add(run_state)
            run_next_state, eval_reward, run_done, _ = eval_env.step(run_action)
            run_reward += eval_reward
            run_state = run_next_state
            run_step += 1

            # prevent evaluation from running forever
            if run_reward >= int(inputs['max_eval_reward']):
                break
        
        end_time = time.perf_counter()
        
        eval_log[round, eval_run, eval, 0] = end_time - start_time
        eval_log[round, eval_run, eval, 1] = run_reward
        eval_log[round, eval_run, eval, 2] = run_step
        eval_log[round, eval_run, eval, 3:14] = loss
        eval_log[round, eval_run, eval, 14] = logtemp
        eval_log[round, eval_run, eval, 15:19] = loss_params
        eval_log[round, eval_run, eval, 19] = cum_steps
    
        print('{} Episode {}: r/st {:1.0f}/{}'
              .format(datetime.now().strftime('%d %H:%M:%S'), eval, run_reward, run_step))

    run = eval_log[round, eval_run, :, 1]
    mean_run = np.mean(run)
    med_run = np.median(run)
    mad_run = np.mean(np.abs(run - mean_run))
    std_run = np.std(run, ddof=0)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    med_step = np.median(step)
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    stats = [mean_run, mean_step, med_run, med_step, mad_run, mad_step, std_run, std_step]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(eval_log[round, eval_run, :, 0])

    print("{} Evaluations Summary {:1.0f}/s r/st: mean {:1.0f}/{:1.0f}, med {:1.0f}/{:1.0f}, mad {:1.0f}/{:1.0f}, std {:1.0f}/{:1.0f}"
          .format(datetime.now().strftime('%d %H:%M:%S'), steps_sec, 
                  stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7]))

def eval_multiplicative(n_gambles: int, agent: object, inputs: dict, eval_log: np.ndarray, eval_risk_log: np.ndarray, 
                        cum_steps: int, round: int, eval_run: int, loss: List[np.ndarray], logtemp: np.ndarray, 
                        loss_params: List[np.ndarray]) \
         -> NoReturn:
    """
    Evaluates agent policy on environment without learning for a fixed number of episodes.

    Parameters:
        n_gambles: number of simultaneous identical gambles
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of existing evalaution results
        eval_risk_log: array of exiting evalaution risk results
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    print('{} {}-{}-{}-{} {} Evaluations at cst {}: C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}'
            .format(datetime.now().strftime('%d %H:%M:%S'), 
                    inputs['algo'], inputs['s_dist'], inputs['loss_fn'], round+1, int(inputs['n_eval']), cum_steps,
                    np.mean(loss[0:2]), np.mean(loss[4:6]), np.mean(loss[6:8]), 
                    np.mean(loss[8:10]), np.mean(loss_params[0:2]), np.mean(loss_params[2:4]), loss[8]+3, np.exp(logtemp)))

    if inputs['ENV_KEY'] <= 16:
        eval_env = eval('coin_flip_envs.'+inputs['env_id'][:9]+'(n_gambles)')
    elif inputs['ENV_KEY'] <= 19:
        eval_env = eval('dice_roll_envs.'+inputs['env_id'][:9]+'(n_gambles)')
    elif inputs['ENV_KEY'] <= 22:
        eval_env = eval('gbm_envs.'+inputs['env_id'][:8]+'(n_gambles)')
    elif inputs['ENV_KEY'] <= 25:
        eval_env = eval('gbm_d_envs.'+inputs['env_id'][:10]+'(n_gambles)')
    else:
        eval_env = eval('dice_roll_sh_envs.'+inputs['env_id'][:-3]+'()')

    for eval_epis in range(int(inputs['n_eval'])):
        start_time = time.perf_counter()
        run_state = eval_env.reset()
        run_done, run_step, run_reward = False, 0, 0

        while not run_done:
            run_action = agent.eval_next_action_mul(run_state)
            run_next_state, eval_reward, done_flags, run_risk = eval_env.step(run_action)
            run_done = done_flags[0]

            run_reward = eval_reward
            run_state = run_next_state
            run_step += 1

            if run_step == int(inputs['max_eval_steps']):
                break
        
        end_time = time.perf_counter()
        
        eval_log[round, eval_run, eval_epis, 0] = end_time - start_time
        eval_log[round, eval_run, eval_epis, 1] = run_reward
        eval_log[round, eval_run, eval_epis, 2] = run_step
        eval_log[round, eval_run, eval_epis, 3:14] = loss
        eval_log[round, eval_run, eval_epis, 14] = logtemp
        eval_log[round, eval_run, eval_epis, 15:19] = loss_params
        eval_log[round, eval_run, eval_epis, 19] = cum_steps
        eval_risk_log[round, eval_run, eval_epis, :] = run_risk

    run = eval_log[round, eval_run, :, 1]
    mean_run = np.mean(run)
    med_run = np.median(run)
    mad_run = np.mean(np.abs(run - mean_run))
    std_run = np.std(run, ddof=0)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    med_step = np.median(step)
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    lev = eval_risk_log[round, eval_run, :, 3]
    mean_lev = np.mean(lev)
    med_lev = np.median(lev)
    mad_lev = np.mean(np.abs(lev - mean_lev))
    std_lev = np.std(lev, ddof=0)

    stats = [(mean_run-1)*100, (med_run-1)*100, mad_run*100, std_run*100,
             mean_lev*100, med_lev*100, mad_lev*100, std_lev*100, 
             mean_step, med_step, mad_step, std_step]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(eval_log[round, eval_run, :, 0])

    print("{} Evaluations Summary {:1.0f}/s mean/med/mad/std: g% {:1.1f}/{:1.1f}/{:1.0f}/{:1.0f} l% {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}"

          .format(datetime.now().strftime('%d %H:%M:%S'), steps_sec, 
                  stats[0], stats[1], stats[2], stats[3], 
                  stats[4], stats[5], stats[6], stats[7],
                  stats[8], stats[9], stats[10], stats[11]))

def eval_market(market_data: np.ndarray, obs_days: int, eval_start_idx: int, agent: object, inputs: dict, 
                eval_log: np.ndarray, eval_risk_log: np.ndarray, cum_steps: int, round: int, 
                eval_run: int, loss: List[np.ndarray], logtemp: np.ndarray, loss_params: List[np.ndarray]) \
        -> NoReturn:
    """
    Evaluates agent policy on environment without learning for a fixed number of episodes.

    Parameters:
        market_data: extracted time sliced data from complete time series
        obs_days: number of previous days agent uses for decision-making
        eval_start_idx: starting index for evaluation episodes without gap 
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of existing evalaution results
        eval_risk_log: array of exiting evalaution risk results
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    print('{} {} Evaluations at cst {}: C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}'
            .format(datetime.now().strftime('%d %H:%M:%S'), 
                    int(inputs['n_eval']), cum_steps,
                    np.mean(loss[0:2]), np.mean(loss[4:6]), np.mean(loss[6:8]), 
                    np.mean(loss[8:10]), np.mean(loss_params[0:2]), np.mean(loss_params[2:4]), loss[8]+3, np.exp(logtemp)))
                        
    n_assets = market_data.shape[1]
    test_length = int(inputs['test_days'] + obs_days - 1)

    if obs_days == 1:
        eval_env = eval('market_envs.Market_'+inputs['env_id'][-7:]+'(n_assets, test_length, obs_days)')
    else:
        eval_env = eval('market_envs.Market_'+inputs['env_id'][-7:-3]+'_Dx'+'(n_assets, test_length, obs_days)')
    
    gap = np.random.randint(int(inputs['gap_days_min']), int(inputs['gap_days_max']) + 1, size=int(inputs['n_eval']))
    gap += eval_start_idx

    for eval_epis in range(int(inputs['n_eval'])):
        start_time = time.perf_counter()

        eval_gap = gap[eval_epis]
        market_slice  = market_data[eval_gap:eval_gap + test_length + 1]
        market_extract = utils.shuffle_data(market_slice, inputs['test_shuffle_days'])

        time_step = 0
        obs_state = utils.observed_market_state(market_extract, time_step, obs_days)

        run_state = eval_env.reset(obs_state)
        run_done, run_step, run_reward = False, 0, 0

        while not run_done:
            time_step += 1
            run_action = agent.eval_next_action_mul(run_state)

            obs_state = utils.observed_market_state(market_extract, time_step, obs_days)

            run_next_state, eval_reward, done_flags, run_risk = eval_env.step(run_action, obs_state)
            run_done = done_flags[0]

            run_reward = eval_reward
            run_state = run_next_state
            run_step += 1
      
        end_time = time.perf_counter()
        
        eval_log[round, eval_run, eval_epis, 0] = end_time - start_time
        eval_log[round, eval_run, eval_epis, 1] = run_reward
        eval_log[round, eval_run, eval_epis, 2] = run_step
        eval_log[round, eval_run, eval_epis, 3:14] = loss
        eval_log[round, eval_run, eval_epis, 14] = logtemp
        eval_log[round, eval_run, eval_epis, 15:19] = loss_params
        eval_log[round, eval_run, eval_epis, 19] = cum_steps
        eval_risk_log[round, eval_run, eval_epis, :] = run_risk
    
    run = eval_log[round, eval_run, :, 1]**252    # convert to annualised (252 day year)
    mean_run = np.mean(run)
    med_run = np.median(run)
    mad_run = np.mean(np.abs(run - mean_run))
    std_run = np.std(run, ddof=0)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    med_step = np.median(step)
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    lev = eval_risk_log[round, eval_run, :, 3]
    mean_lev = np.mean(lev)
    med_lev = np.median(lev)
    mad_lev = np.mean(np.abs(lev - mean_lev))
    std_lev = np.std(lev, ddof=0)

    stats = [(mean_run-1)*100, (med_run-1)*100, mad_run*100, std_run*100,
             mean_lev*100, med_lev*100, mad_lev*100, std_lev*100, 
             mean_step, med_step, mad_step, std_step]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(eval_log[round, eval_run, :, 0])

    print("{} Evaluations Summary {:1.0f}/s mean/med/mad/std: g% {:1.1f}/{:1.1f}/{:1.0f}/{:1.0f} l% {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}"

          .format(datetime.now().strftime('%d %H:%M:%S'), steps_sec, 
                  stats[0], stats[1], stats[2], stats[3], 
                  stats[4], stats[5], stats[6], stats[7],
                  stats[8], stats[9], stats[10], stats[11]))