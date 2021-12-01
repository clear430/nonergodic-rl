#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  rl_market.py
python version:         3.9
torch verison:          1.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for performing agent training in market environments.
"""

import sys
sys.path.append("./")

from datetime import datetime
import numpy as np
import os
import time
from typing import NoReturn

from algos.algo_sac import Agent_sac
from algos.algo_td3 import Agent_td3
import envs.market_envs as market_envs
import extras.plots_summary as plots
import extras.eval_episodes as eval_episodes
import extras.utils as utils

def market_env(gym_envs: dict, inputs: dict, market_data: np.ndarray, obs_days: int) -> NoReturn:
    """
    Conduct experiments for market environments.

    Parameters:
        gym_envs: all environment details
        inputs: all training and evaluation details 
        market_data: extracted time sliced data from complete time series
        obs_days: number of previous days agent uses for decision-making
    """
    n_assets = market_data.shape[1]
    train_length = int(inputs['train_days'] + obs_days - 1)
    test_length = int(inputs['test_days'] + obs_days - 1)
    gap_max = int(inputs['gap_days_max'])

    sample_length = int(train_length + test_length + gap_max + obs_days - 1)

    obs_days_str = '_D' + str(obs_days)
    inputs: dict= {'env_id': gym_envs[str(inputs['ENV_KEY'])][0] + obs_days_str, **inputs}
    if obs_days == 1:
        env = eval('market_envs.Market_'+gym_envs[str(inputs['ENV_KEY'])][0][-4:]+'_D1'+'(n_assets, train_length, obs_days)')
    else:
        env = eval('market_envs.Market_'+gym_envs[str(inputs['ENV_KEY'])][0][-4:]+'_Dx'+'(n_assets, train_length, obs_days)')

    inputs: dict = {
        'input_dims': env.observation_space.shape, 'num_actions':  env.action_space.shape[0], 
        'max_action': env.action_space.high.min(), 'min_action': env.action_space.low.max(),    # assume all actions span equal domain 
        'random': gym_envs[str(inputs['ENV_KEY'])][3], 'dynamics': 'MKT',    # gambling dynamics 'MKT' (market)
        'n_trials': inputs['n_trials_mar'], 'n_cumsteps': int(inputs['n_cumsteps_mar']),
        'eval_freq': inputs['eval_freq_mar'], 'n_eval': inputs['n_eval_mar'], 
        'algo': 'TD3', 'loss_fn': 'MSE', 'multi_steps': 1, **inputs
        }

    risk_dim = utils.market_log_dim(inputs, n_assets)
    
    for algo in inputs['algo_name']:
        for loss_fn in inputs['critic_loss']:
            for mstep in inputs['bootstraps']:

                inputs['loss_fn'], inputs['algo'], inputs['multi_steps'] = loss_fn, algo, mstep

                trial_log = np.zeros((inputs['n_trials'], int(inputs['n_cumsteps']), 19))
                eval_log = np.zeros((inputs['n_trials'], int(inputs['n_cumsteps'] / inputs['eval_freq']), int(inputs['n_eval']), 20))
                directory = utils.save_directory(inputs, results=True)

                trial_risk_log = np.zeros((inputs['n_trials'], int(inputs['n_cumsteps']), risk_dim))
                eval_risk_log = np.zeros((inputs['n_trials'], int(inputs['n_cumsteps'] / inputs['eval_freq']), int(inputs['n_eval']), risk_dim))

                for round in range(inputs['n_trials']):

                    time_log, score_log, step_log, logtemp_log, loss_log, loss_params_log = [], [], [], [], [], []
                    risk_log = []
                    cum_steps, eval_run, episode = 0, 0, 1
                    best_score = env.reward_range[0]
                    if inputs['continue'] == True:
                        inputs['initial_logtemp'] = logtemp if round > 1 else False    # load existing SAC parameter to continue learning

                    agent = Agent_td3(inputs) if inputs['algo'] == 'TD3' else Agent_sac(inputs)
                    if inputs['continue'] == True:
                        agent.load_models() if round > 1 else False    # load existing actor-critic parameters to continue learning

                    while cum_steps < int(inputs['n_cumsteps']):
                        start_time = time.perf_counter()

                        # randomly extract sequential time series from history and shuffle
                        market_slice, start_idx = utils.time_slice(market_data, train_length, sample_length)
                        market_extract = utils.shuffle_data(market_slice, inputs['train_shuffle_days'])

                        time_step = 0
                        obs_state = utils.observed_market_state(market_extract, time_step, obs_days)

                        state = env.reset(obs_state)
                        done, step, score = False, 0, 0

                        while not done:
                            time_step += 1

                            if step >= int(inputs['random']):
                                action = agent.select_next_action(state)
                            else:
                                # take random actions during initial warmup period to generate new seed
                                action = env.action_space.sample()

                            obs_state = utils.observed_market_state(market_extract, time_step, obs_days)

                            next_state, reward, env_done, risk = env.step(action, obs_state)
                            done, learn_done = env_done[0], env_done[1]    # environment done flags
                            
                            agent.store_transistion(state, action, reward, next_state, learn_done)

                            # gradient update interval (perform backpropagation)
                            if cum_steps %  int(inputs['grad_step'][inputs['algo']]) == 0:
                                loss, logtemp, loss_params = agent.learn()

                            state = next_state
                            score = reward
                            step += 1
                            cum_steps += 1
                            end_time = time.perf_counter()

                            # conduct periodic agent evaluation episodes without learning
                            if cum_steps % int(inputs['eval_freq']) == 0:

                                eval_start_idx = start_idx + step
                                eval_episodes.eval_market(market_data, obs_days, eval_start_idx, agent, inputs, 
                                                          eval_log, eval_risk_log, cum_steps, round, eval_run, 
                                                          loss, logtemp, loss_params)
                                eval_run += 1

                            if cum_steps > int(inputs['n_cumsteps']-1):
                                break

                        time_log.append(end_time - start_time)
                        score_log.append(score)
                        step_log.append(step)
                        loss_log.append(loss)
                        logtemp_log.append(logtemp)
                        loss_params_log.append(loss_params)
                        risk_log.append(risk)

                        # save actor-critic neural network weights for checkpointing
                        trail_score = np.mean(score_log[-inputs['trail']:])
                        if trail_score > best_score:
                            best_score = trail_score
                            agent.save_models()
                            print('New high trailing score!')

                        print('{} {}-{}-{}-{} ep/st/cst {}/{}/{} {:1.0f}/s: g_pa/V {:1.1f}%/${:1.1f}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}'
                                .format(datetime.now().strftime('%d %H:%M:%S'),
                                        inputs['algo'], inputs['s_dist'], inputs['loss_fn'], round+1, episode, step, cum_steps, step/time_log[-1], 
                                        (reward**252-1)*100, risk[1], np.mean(loss[0:2]), np.mean(loss[4:6]), np.mean(loss[6:8]), 
                                        np.mean(loss[8:10]), np.mean(loss_params[0:2]), np.mean(loss_params[2:4]), loss[8]+3, np.exp(logtemp)))
                        
                        # EPISODE PRINT STATEMENT
                                        # date time,
                                        # rl_algorithm-sampling_distribution-loss_function-trial,  ep/st/cst = episode/steps/cumulative_steps,  /s = training_steps_per_second,
                                        # Vg/V = annualised-time-average-growth-rate/valuation,  C/Cm/Cs = avg_critic_loss/max_critic_loss/shadow_critic_loss
                                        # c/k/a/A/T = avg_Cauchy_scale/avg_CIM_kernel_size/avg_tail_exponent/avg_actor_loss/sac_entropy_temperature

                        episode += 1

                    count = len(score_log)

                    trial_log[round, :count, 0], trial_log[round, :count, 1] = time_log, score_log
                    trial_log[round, :count, 2], trial_log[round, :count, 3:14] = step_log, loss_log
                    trial_log[round, :count, 14], trial_log[round, :count, 15:] = logtemp_log, loss_params_log
                    trial_risk_log[round, :count, :] = risk_log

                    if not os.path.exists('./results/market/'+inputs['env_id']):
                        os.makedirs('./results/market/'+inputs['env_id'])

                    if inputs['n_trials'] == 1:
                        plots.plot_learning_curve(inputs, trial_log[round], directory+'.png')

                # truncate training trial log arrays up to maximum episodes
                count_episodes = [np.min(np.where(trial_log[trial, :, 0] == 0)) for trial in range(int(inputs['n_trials']))]
                max_episode = np.max(count_episodes) 
                trial_log, trial_risk_log = trial_log[:, :max_episode, :], trial_risk_log[:, :max_episode, :]

                np.save(directory+'_trial.npy', trial_log)
                np.save(directory+'_eval.npy', eval_log)
                np.save(directory+'_trial_risk.npy', trial_risk_log)
                np.save(directory+'_eval_risk.npy', eval_risk_log)

                if inputs['n_trials'] > 1:
                    plots.plot_eval_loss_2d_multi(inputs, eval_log, directory+'_2d.png')    # plot of agent evaluation round scores and training critic losses across all trials
                    plots.plot_trial_curve(inputs, trial_log, directory+'_trial.png')       # plot of agent training with linear interpolation across all trials