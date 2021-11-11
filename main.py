#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  main.py
usage:                  python main.py
python version:         3.9
torch verison:          1.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for executing all agent training and conducting tests on provided inputs.

Instructions:
    1. Select algorithms, critic loss functions, multi-steps, and environments using available options
       and enter them into the provided four lists.
    2. Modify inputs dictionary containing training parameters and model hyperparameters if required.
    3. Run python file and upon completion all learned parameters will be placed into ./models/env_id 
       and data and plots regarding training into ./results/env_id directories.
"""

import time
from typing import Dict, List

from algos.algo_sac import Agent_sac
from algos.networks_sac import ActorNetwork as ActorNetwork_sac
from algos.networks_sac import CriticNetwork as CriticNetwork_sac
from algos.algo_td3 import Agent_td3
from algos.networks_td3 import ActorNetwork as ActorNetwork_td3
from algos.networks_td3 import CriticNetwork as CriticNetwork_td3
from extras.replay import ReplayBuffer
from extras.replay_torch import ReplayBufferTorch
from scripts.rl_additive import additive_env
from scripts.rl_market import market_env
from scripts.rl_multiplicative import multiplicative_env

# model-free off-policy agents: list ['SAC', 'TD3']
algo_name: List[str] = ['TD3']

# critic loss functions: list ['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'CIM', 'MSE2', 'MSE4', 'MSE6']
critic_loss: List[str] = ['MSE']

# bootstrapping of target critic values and discounted rewards: list [integer > 0] 
multi_steps: List[int] = [1]

# environments to train agent: list [integer ENV_KEY from gym_envs]
envs: List[int] = [61]

gym_envs: Dict[str, list] = {
    # ENV_KEY: [env_id, state_dim, action_dim, intial warm-up steps to generate random seed]

    # ADDITIVE ENVIRONMENTS

    # OpenAI Box2D continuous control tasks
    '0': ['LunarLanderContinuous-v2', 8, 2, 1e3], 
    '1': ['BipedalWalker-v3', 24, 4, 1e3],
    '2': ['BipedalWalkerHardcore-v3', 24, 4, 1e3],
    # Roboschool environments ported to PyBullet
    '3': ['CartPoleContinuousBulletEnv-v0', 4, 1, 1e3], 
    '4': ['InvertedPendulumBulletEnv-v0', 5, 1, 1e3],
    '5': ['InvertedDoublePendulumBulletEnv-v0', 9, 1, 1e3], 
    '6': ['HopperBulletEnv-v0', 15, 3, 1e3], 
    '7': ['Walker2DBulletEnv-v0', 22, 6, 1e3],
    '8': ['HalfCheetahBulletEnv-v0', 26, 6, 1e4],
    '9': ['AntBulletEnv-v0', 28, 8, 1e4],
    '10': ['HumanoidBulletEnv-v0', 44, 17, 1e4], 
    # KOD*LAB quadruped direct-drive legged robots ported to PyBullet
    '11': ['MinitaurBulletEnv-v0', 28, 8, 1e4],
    # DeepMimic simulation of a imitating Humanoid mimic ported to PyBullet
    '12': ['HumanoidDeepMimicWalkBulletEnv-v1', 197, 36, 1e4],
    '13': ['HumanoidDeepMimicBackflipBulletEnv-v1', 197, 36, 1e4],

    # MULTIPLICATVE ENVIRONMENTS

    # assets following the equally likely +50%/-40% gamble
    '14': ['Coin_n1_InvA', 2, 1, 1e3], '15': ['Coin_n2_InvA', 3, 2, 1e3], '16': ['Coin_n10_InvA', 11, 10, 1e3],
    '17': ['Coin_n1_InvB', 2, 2, 1e3], '18': ['Coin_n2_InvB', 3, 3, 1e3], '19': ['Coin_n10_InvB', 11, 11, 1e3],
    '20': ['Coin_n1_InvC', 2, 3, 1e3], '21': ['Coin_n2_InvC', 3, 4, 1e3], '22': ['Coin_n10_InvC', 11, 12, 1e3],
    # assets following the dice roll
    '23': ['Dice_n1_InvA', 2, 1, 1e3], '24': ['Dice_n2_InvA', 3, 2, 1e3], '25': ['Dice_n10_InvA', 11, 10, 1e3],
    '26': ['Dice_n1_InvB', 2, 2, 1e3], '27': ['Dice_n2_InvB', 3, 3, 1e3], '28': ['Dice_n10_InvB', 11, 11, 1e3],
    '29': ['Dice_n1_InvC', 2, 3, 1e3], '30': ['Dice_n2_InvC', 3, 4, 1e3], '31': ['Dice_n10_InvC', 11, 12, 1e3],
    # assets following GBM
    '32': ['GBM_n1_InvA', 2, 1, 1e3], '33': ['GBM_n2_InvA', 3, 2, 1e3], '34': ['GBM_n10_InvA', 11, 10, 1e3],
    '35': ['GBM_n1_InvB', 2, 2, 1e3], '36': ['GBM_n2_InvB', 3, 3, 1e3], '37': ['GBM_n10_InvB', 11, 11, 1e3],
    '38': ['GBM_n1_InvC', 2, 3, 1e3], '39': ['GBM_n2_InvC', 3, 4, 1e3], '40': ['GBM_n10_InvC', 11, 12, 1e3],
    # assets following GBM with discrete compounding
    '41': ['GBM_D_n1_InvA', 2, 1, 1e3], '42': ['GBM_D_n2_InvA', 3, 2, 1e3], '43': ['GBM_D_n10_InvA', 11, 10, 1e3],
    '44': ['GBM_D_n1_InvB', 2, 2, 1e3], '45': ['GBM_D_n2_InvB', 3, 3, 1e3], '46': ['GBM_D_n10_InvB', 11, 11, 1e3],
    '47': ['GBM_D_n1_InvC', 2, 3, 1e3], '48': ['GBM_D_n2_InvC', 3, 4, 1e3], '49': ['GBM_D_n10_InvC', 11, 12, 1e3],
    # assets following dice roll without (U) and with insurance (I) safe haven
    '50': ['Dice_SH_n1_U', 2, 1, 1e3], '51': ['Dice_SH_n1_I', 2, 1, 1e3], 
    '52': ['Dice_SH_n1_InvA_U', 2, 1, 1e3], '53': ['Dice_SH_n1_InvA_I', 3, 2, 1e3],
    '54': ['Dice_SH_n1_InvB_U', 2, 2, 1e3], '55': ['Dice_SH_n1_InvB_I', 3, 3, 1e3],  
    '56': ['Dice_SH_n1_InvC_U', 2, 3, 1e3], '57': ['Dice_SH_n1_InvC_I', 3, 4, 1e3], 

    # MARKET ENVIRONMENTS
    # Stooq-based (cleaned) historical daily data from 1985-10-01 to 2021-11-10

    # S&P500 index (^SPX)
    '58': ['SNP_InvA', 2, 1, 1e3], '59': ['SNP_InvB', 2, 2, 1e3], '60': ['SNP_InvC', 2, 3, 1e3],
    # US equity indicies (^SPX, ^DJI, ^NDX)
    '61': ['EI_InvA', 4, 3, 1e3], '62': ['EI_InvB', 4, 4, 1e3], '63': ['EI_InvC', 4, 5, 1e3],
    # US-listed equity indicies and a few commoditites
    '64': ['Minor_InvA', 7, 6, 1e3], '65': ['Minor_InvB', 7, 7, 1e3], '66': ['Minor_InvC', 7, 8, 1e3],
    # US-listed equity indicies and several commoditites
    '67': ['Medium_InvA', 10, 9, 1e3], '68': ['Medium_InvB', 10, 10, 1e3], '69': ['Medium_InvC', 10, 11, 1e3],
    # US-listed equity indicies and many commoditites
    '70': ['Major_InvA', 15, 14, 1e3], '71': ['Major_InvB', 15, 15, 1e3], '72': ['Major_InvC', 15, 16, 1e3],
    # US equity indicies and 26/30 Dow Jones (^DJI) components
    '73': ['DJI_InvA', 30, 29, 1e3], '74': ['DJI_InvB', 30, 30, 1e3], '75': ['DJI_InvC', 30, 31, 1e3],
    # Combined Major + DJI market
    '76': ['Full_InvA', 41, 40, 1e3], '77': ['Full_InvB', 41, 41, 1e3], '78': ['Full_InvC', 41, 42, 1e3],
    }

inputs_dict: dict = {
    # additive environment execution parameters
    'n_trials_add': 10,                         # number of total unique training trials
    'n_cumsteps_add': 3e5,                      # maximum cumulative steps per trial (must be greater than environment warm-up)
    'eval_freq_add': 1e3,                       # interval of steps between evaluation episodes
    'n_eval_add': 1e1,                          # number of evalution episodes
    'max_eval_reward': 1e4,                     # maximum score per evaluation episode

    # multiplicative environment execution parameters
    'n_trials_mul': 10,                         # ibid.
    'n_cumsteps_mul': 8e4,                      # ibid.
    'eval_freq_mul': 1e3,                       # ibid.
    'n_eval_mul': 1e3,                          # ibid.
    'max_eval_steps': 1e0,                      # maximum steps per evaluation episode

    # multiplicative environment execution parameters
    'n_trials_mar': 10,                         # ibid.
    'n_cumsteps_mar': 8e4,                      # ibid.
    'eval_freq_mar': 1e3,                       # ibid.
    'n_eval_mar': 1e2,                          # ibid.
    'train_years': 3,                           # length of sequential training periods
    'test_years': 1,                            # length of sequential testing periods
    'gap_years': 0.25,                          # length between end of training and start of testing periods
    'train_shuffle_days': 10,                   # size of interval to shuffle time-series data for training
    'test_shuffle_days': 5,                     # size of interval to shuffle time-series data for inference

    # learning variables
    'buffer': 1e6,                              # maximum transistions in experience replay buffer
    'buffer_gpu': True,                         # use GPU-accelerated buffer (faster for single-step, much slower for multi-step)
    'discount': 0.99,                           # discount factor for successive steps
    'trail': 50,                                # moving average of training episode scores used for model saving
    'cauchy_scale': 1,                          # Cauchy scale parameter initialisation value
    'actor_percentile': 1,                      # bottom percentile of actor mini-batch to be maximised (>0, <=1)
    'r_abs_zero': None,                         # defined absolute zero value for rewards
    'continue': False,                          # whether to continue learning with same parameters across trials

    # critic loss aggregation
    'critic_mean_type': 'E',                    # critic network learning either empirical 'E' or shadow 'S' (only E) 
    'shadow_low_mul': 1e0,                      # lower bound multiplier of minimum for critic power law  
    'shadow_high_mul': 1e1,                     # upper bound multiplier of maximum for critic power law

    # SAC hyperparameters (https://arxiv.org/pdf/1812.05905.pdf)
    'sac_actor_learn_rate': 3e-4,               # actor learning rate (Adam optimiser)
    'sac_critic_learn_rate': 3e-4,              # critic learning rate (Adam optimiser)
    'sac_temp_learn_rate': 3e-4,                # log temperature learning rate (Adam optimiser)
    'sac_layer_1_units': 256,                   # nodes in first fully connected layer
    'sac_layer_2_units': 256,                   # nodes in second fully connected layer
    'sac_actor_step_update': 1,                 # actor policy network update frequency (steps)
    'sac_temp_step_update': 1,                  # temperature update frequency (steps)
    'sac_target_critic_update': 1,              # target critic networks update frequency (steps)
    'initial_logtemp': 0,                       # log weighting given to entropy maximisation
    'reward_scale': 1,                          # constant scaling factor of next reward ('inverse temperature')
    'reparam_noise': 1e-6,                      # miniscule constant to keep logarithm bounded

    # TD3 hyperparameters (https://arxiv.org/pdf/1802.09477.pdf)          
    'td3_actor_learn_rate': 1e-3,               # ibid.
    'td3_critic_learn_rate': 1e-3,              # ibid.
    'td3_layer_1_units': 400,                   # ibid.
    'td3_layer_2_units': 300,                   # ibid.
    'td3_actor_step_update': 2,                 # ibid.
    'td3_target_actor_update': 2,               # target actor network update frequency (steps)
    'td3_target_critic_update': 2,              # ibid.
    'policy_noise': 0.1,                        # Gaussian exploration noise added to next actions
    'target_policy_noise': 0.2,                 # Gaussian noise added to next target actions as a regulariser
    'target_policy_clip': 0.5,                  # Clipping of Gaussian noise added to next target actions

    # shared parameters
    'target_update_rate': 5e-3,                 # Polyak averaging rate for target network parameter updates
    's_dist': 'N',                              # actor policy sampling via 'L' (Laplace) or 'N' (Normal) distribution 
    'batch_size': {'SAC': 256, 'TD3': 100},     # mini-batch size
    'grad_step': {'SAC': 1, 'TD3': 1},          # standard gradient update frequency (steps)

    # environment details
    'algo_name': [algo.upper() for algo in algo_name],
    'critic_loss': [loss.upper() for loss in critic_loss],
    'bootstraps': multi_steps,
    'envs': envs,
    'ENV_KEY': 0
    }

# CONDUCT TESTS
gte0: str = 'must be greater than or equal to 0'
gt0: str = 'must be greater than 0'
gte1: str = 'must be greater than or equal to 1'

# execution tests
assert isinstance(inputs_dict['n_trials_add'], (float, int)) and \
    int(inputs_dict['n_trials_add']) >= 1, gte1
assert isinstance(inputs_dict['n_cumsteps_add'], (float, int)) and \
    set(list(str(inputs_dict['n_cumsteps_add'])[2:])).issubset(set(['0', '.'])) and \
        int(inputs_dict['n_cumsteps_add']) >= 1, \
            'must consist of only 2 leading non-zero digits and be greater than or equal to 1'
assert isinstance(inputs_dict['eval_freq_add'], (float, int)) and \
    int(inputs_dict['eval_freq_add']) >= 1 and \
        int(inputs_dict['eval_freq_add']) <= int(inputs_dict['n_cumsteps_add']), \
            'must be greater than or equal to 1 and less than or equal to n_cumsteps'
assert isinstance(inputs_dict['n_eval_add'], (float, int)) and \
    int(inputs_dict['n_eval_add']) >= 1, gte1
assert isinstance(inputs_dict['max_eval_reward'], (float, int)) and \
    inputs_dict['max_eval_reward'] > 0, gt0
assert isinstance(inputs_dict['n_trials_add'], (float, int)) and \
    int(inputs_dict['n_trials_add']) >= 1, gte1
assert isinstance(inputs_dict['n_cumsteps_mul'], (float, int)) and \
    set(list(str(inputs_dict['n_cumsteps_mul'])[2:])).issubset(set(['0', '.'])) and \
        int(inputs_dict['n_cumsteps_mul']) >= 1, \
            'must consist of only 2 leading non-zero digits and be greater than or equal to 1'
assert isinstance(inputs_dict['eval_freq_mul'], (float, int)) and \
    int(inputs_dict['eval_freq_mul']) >= 1 and \
        int(inputs_dict['eval_freq_mul']) <= int(inputs_dict['n_cumsteps_mul']), \
            'must be greater than or equal to 1 and less than or equal to n_cumsteps'
assert isinstance(inputs_dict['n_eval_mul'], (float, int)) and \
    int(inputs_dict['n_eval_mul']) >= 1, gte1
assert isinstance(inputs_dict['max_eval_steps'], (float, int)) and \
    int(inputs_dict['max_eval_steps']) >= 1, gte1
assert isinstance(inputs_dict['n_cumsteps_mar'], (float, int)) and \
    set(list(str(inputs_dict['n_cumsteps_mar'])[2:])).issubset(set(['0', '.'])) and \
        int(inputs_dict['n_cumsteps_mar']) >= 1, \
            'must consist of only 2 leading non-zero digits and be greater than or equal to 1'
assert isinstance(inputs_dict['eval_freq_mar'], (float, int)) and \
    int(inputs_dict['eval_freq_mar']) >= 1 and \
        int(inputs_dict['eval_freq_mar']) <= int(inputs_dict['n_cumsteps_mul']), \
            'must be greater than or equal to 1 and less than or equal to n_cumsteps'
assert isinstance(inputs_dict['n_eval_mar'], (float, int)) and \
    int(inputs_dict['n_eval_mar']) >= 1, gte1
assert isinstance(inputs_dict['train_years'], (float, int)) and \
    int(inputs_dict['train_years']) > 0, gt0
assert isinstance(inputs_dict['test_years'], (float, int)) and \
    int(inputs_dict['test_years']) > 0, gt0
assert isinstance(inputs_dict['gap_years'], (float, int)) and \
    int(inputs_dict['gap_years']) >= 0, gte0
assert isinstance(inputs_dict['train_shuffle_days'], (int)) and \
    int(inputs_dict['train_shuffle_days']) >= 1, gte1
assert isinstance(inputs_dict['test_shuffle_days'], (int)) and \
    int(inputs_dict['test_shuffle_days']) >= 1, gte1

# learning varaible tests
assert isinstance(inputs_dict['buffer'], (float, int)) and \
    set(list(str(inputs_dict['buffer'])[2:])).issubset(set(['0', '.'])) and \
        int(inputs_dict['buffer']) >= 1 and \
            inputs_dict['buffer'] >= inputs_dict['n_cumsteps_add'] and \
                inputs_dict['buffer'] >= inputs_dict['n_cumsteps_mul'], \
                    'must consist of only upto 2 leading non-zero digits and be greater than or equal to both 1 and n_cumsteps'
assert isinstance(inputs_dict['buffer_gpu'], bool), 'must be either True or False'
assert inputs_dict['discount'] >= 0 \
    and inputs_dict['discount'] < 1, 'should be within [0, 1)'
assert isinstance(inputs_dict['trail'], (float, int)) and \
    int(inputs_dict['trail']) >= 1, gte1
assert isinstance(inputs_dict['cauchy_scale'], (float, int)) and \
    inputs_dict['cauchy_scale'] > 0, gt0
assert isinstance(inputs_dict['actor_percentile'], (float, int)) and \
    inputs_dict['actor_percentile'] > 0 and \
        inputs_dict['actor_percentile'] <= 1, 'must be within (0, 1]'
assert isinstance(inputs_dict['r_abs_zero'], (float, int)) or inputs_dict['r_abs_zero'] == None, \
    'either real number or None'
assert isinstance(inputs_dict['continue'], bool), 'must be either True or False'

# critic loss aggregation tests
assert inputs_dict['critic_mean_type'] == 'E' or 'S', 'must be either "E" or "S"'
assert isinstance(inputs_dict['shadow_low_mul'], (float, int)) and \
    inputs_dict['shadow_low_mul'] >= 0, gte0
assert isinstance(inputs_dict['shadow_high_mul'], (float, int)) and \
    inputs_dict['shadow_high_mul'] > 0, gt0

# SAC hyperparameter tests
assert isinstance(inputs_dict['sac_actor_learn_rate'], (float, int)) and \
    inputs_dict['sac_actor_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['sac_critic_learn_rate'], (float, int)) and \
    inputs_dict['sac_critic_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['sac_temp_learn_rate'], (float, int)) and \
    inputs_dict['sac_temp_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['sac_layer_1_units'], (float, int)) and \
    int(inputs_dict['sac_layer_1_units']) >= 1, gte1
assert isinstance(inputs_dict['sac_layer_2_units'], (float, int)) and \
    int(inputs_dict['sac_layer_2_units']) >= 1, gte1
assert isinstance(inputs_dict['sac_actor_step_update'], (float, int)) and \
    int(inputs_dict['sac_actor_step_update']) >= 1, gte1
assert isinstance(inputs_dict['sac_temp_step_update'], (float, int)) and \
    int(inputs_dict['sac_temp_step_update']) >= 1, gte1
assert isinstance(inputs_dict['sac_target_critic_update'], (float, int)) and \
    int(inputs_dict['sac_target_critic_update']) >= 1, gte1
assert isinstance(inputs_dict['initial_logtemp'], (float, int)), 'must be any real number'
assert isinstance(inputs_dict['reparam_noise'], float) and \
    inputs_dict['reparam_noise'] > 1e-7 and \
        inputs_dict['reparam_noise'] < 1e-5, 'must be any real number in the vicinity of 1e-6'

# TD3 hyperparameter tests
assert isinstance(inputs_dict['td3_actor_learn_rate'], (float, int)) and \
    inputs_dict['td3_actor_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['td3_critic_learn_rate'], (float, int)) and \
    inputs_dict['td3_critic_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['td3_layer_1_units'], (float, int)) and \
    int(inputs_dict['td3_layer_1_units']) >= 1, gte1
assert isinstance(inputs_dict['td3_layer_2_units'], (float, int)) and \
    int(inputs_dict['td3_layer_2_units']) >= 1, gte1
assert isinstance(inputs_dict['td3_actor_step_update'], (float, int)) and \
    int(inputs_dict['td3_actor_step_update']) >= 1, gte1
assert isinstance(inputs_dict['td3_target_actor_update'], (float, int)) and \
    int(inputs_dict['td3_target_actor_update']) >= 1, gte1
assert isinstance(inputs_dict['td3_target_critic_update'], (float, int)) and \
    int(inputs_dict['td3_target_critic_update']) >= 1, gte1
assert isinstance(inputs_dict['td3_target_critic_update'], (float, int)) and \
    int(inputs_dict['td3_target_critic_update']) >= 1, gte1
assert isinstance(inputs_dict['policy_noise'], (float, int)) and \
    inputs_dict['policy_noise'] >= 0, gte0
assert isinstance(inputs_dict['target_policy_noise'], (float, int)) and \
    inputs_dict['target_policy_noise'] >= 0, gte0
assert isinstance(inputs_dict['target_policy_clip'], (float, int)) and \
    inputs_dict['target_policy_clip'] >= 0, gte0

# shared parameter tests
assert isinstance(inputs_dict['target_update_rate'], (float, int)) and \
    inputs_dict['target_update_rate'] > 0, gt0
assert inputs_dict['s_dist'] == ('N' or 'L') or \
    (inputs_dict['algo_name'][0] == 'SAC' and inputs_dict['s_dist'] == 'MVN'), \
        'must be either "N", "S" or "MVN" (only for SAC)'
assert isinstance(inputs_dict['batch_size'], dict) and \
    isinstance(inputs_dict['batch_size']['TD3'], (float, int)) and \
        int(inputs_dict['batch_size']['TD3']) >= 1 and \
            isinstance(inputs_dict['batch_size']['SAC'], (float, int)) and \
                int(inputs_dict['batch_size']['SAC']) >= 1, \
                    'mini-batch sizes must be at least 1 for all algorithms'
assert isinstance(inputs_dict['grad_step'], dict) and \
    isinstance(inputs_dict['grad_step']['TD3'], (float, int)) and \
        int(inputs_dict['grad_step']['TD3']) >= 1 and \
            isinstance(inputs_dict['grad_step']['SAC'], (float, int)) and \
                int(inputs_dict['grad_step']['SAC']) >= 1, \
                    'gradient step must be at least 1 for all algorithms'

# environment tests
assert isinstance(inputs_dict['algo_name'], list) and \
    set(inputs_dict['algo_name']).issubset(set(['SAC', 'TD3'])), \
        'algorithms must be a list containing "SAC" and/or "TD3"'
assert isinstance(inputs_dict['critic_loss'], list) and \
    set(inputs_dict['critic_loss']).issubset(set(['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'CIM', 'MSE2', 'MSE4', 'MSE6'])), \
        'critic losses must be a list containing "MSE", "HUB", "MAE", "HSC", "CAU", "TCAU", "CIM", "MSE2", "MSE4", and/or "MSE6"'
assert isinstance(inputs_dict['bootstraps'], list) and \
    all(isinstance(mstep, int) for mstep in inputs_dict['bootstraps']) and \
        all(mstep >= 1 for mstep in inputs_dict['bootstraps']), \
            'multi-steps must be a list of positve integers'

keys: List[int] = [int(env) for env in gym_envs]
assert isinstance(inputs_dict['envs'], list) and \
    set(inputs_dict['envs']).issubset(set(keys)), \
        'environments must be selected from gym_envs dict keys'
for key in inputs_dict['envs']:
    assert isinstance(gym_envs[str(key)], list) and \
        isinstance(gym_envs[str(key)][0], str) and \
            all(isinstance(x, (float, int)) for x in gym_envs[str(key)][1:]), \
                'environment {} details must be a list of the form [string, real, real, real]'.format(key)
    assert int(gym_envs[str(key)][1]) >= 1, 'environment {} must have at least one state'.format(key)
    assert int(gym_envs[str(key)][2]) >= 1, 'environment {} must have at least one action'.format(key)
    if key <= 13:
        assert int(gym_envs[str(key)][3]) >= 0 and \
            int(gym_envs[str(key)][3]) <= int(inputs_dict['n_cumsteps_add']), \
                'environment {} warm-up must be less than or equal to total training steps'.format(key)
    else:
        assert int(gym_envs[str(key)][3]) >= 0 and \
            int(gym_envs[str(key)][3]) <= int(inputs_dict['n_cumsteps_mul']), \
                'environment {} warm-up must be less than or equal to total training steps'.format(key)

# SAC algorithm method checks
assert hasattr(Agent_sac, 'select_next_action'), 'missing SAC agent action selection'
assert hasattr(Agent_sac, 'eval_next_action'), 'missing SAC agent evaluation action selection'
assert hasattr(Agent_sac, 'store_transistion'), 'missing SAC transition storage functionality'
assert hasattr(Agent_sac, 'learn'), 'missing SAC agent learning functionality'
assert hasattr(Agent_sac, 'save_models'), 'missing SAC agent save functionality'
assert hasattr(Agent_sac, 'load_models'), 'missing SAC agent load functionality'
assert hasattr(ActorNetwork_sac, 'stochastic_uv'), 'missing SAC univariate sampling'
assert hasattr(ActorNetwork_sac, 'stochastic_mv_gaussian'), 'missing SAC multi-variate Gaussian sampling'
assert hasattr(ActorNetwork_sac, 'save_checkpoint'), 'missing SAC actor saving functionality'
assert hasattr(ActorNetwork_sac, 'load_checkpoint'), 'missing SAC actor load functionality'
assert hasattr(CriticNetwork_sac, 'forward'), 'missing SAC critic forward propagation'
assert hasattr(CriticNetwork_sac, 'save_checkpoint'), 'missing SAC critic saving functionality'
assert hasattr(CriticNetwork_sac, 'load_checkpoint'), 'missing SAC critic load functionality'

# TD3 algorithm method checks
assert hasattr(Agent_td3, 'select_next_action'), 'missing TD3 agent action selection'
assert hasattr(Agent_td3, 'eval_next_action'), 'missing TD3 agent evaluation action selection'
assert hasattr(Agent_td3, 'store_transistion'), 'missing TD3 transition storage functionality'
assert hasattr(Agent_td3, 'learn'), 'missing TD3 agent learning functionality'
assert hasattr(Agent_td3, 'save_models'), 'missing TD3 agent save functionality'
assert hasattr(Agent_td3, 'load_models'), 'missing TD3 agent load functionality'
assert hasattr(ActorNetwork_td3, 'forward'), 'missing TD3 actor forward propagation'
assert hasattr(ActorNetwork_td3, 'save_checkpoint'), 'missing TD3 actor saving functionality'
assert hasattr(ActorNetwork_td3, 'load_checkpoint'), 'missing TD3 actor load functionality'
assert hasattr(CriticNetwork_td3, 'forward'), 'missing TD3 critic forward propagation'
assert hasattr(CriticNetwork_td3, 'save_checkpoint'), 'missing TD3 critic saving functionality'
assert hasattr(CriticNetwork_td3, 'load_checkpoint'), 'missing TD3 critic load functionality'

# replay buffer method checks
assert hasattr(ReplayBuffer, 'store_exp'), 'missing transition store functionality'
assert hasattr(ReplayBuffer, 'sample_exp'), 'missing uniform transition sampling functionality'
assert hasattr(ReplayBufferTorch, 'store_exp'), 'missing transition store functionality'
assert hasattr(ReplayBufferTorch, 'sample_exp'), 'missing uniform transition sampling functionality'

if __name__ == '__main__':

    start_time = time.perf_counter()

    for env_key in envs:
        inputs_dict['ENV_KEY'] = env_key 
            
        if env_key <= 13:
            additive_env(gym_envs=gym_envs, inputs=inputs_dict)
        elif env_key <= 57:
            multiplicative_env(gym_envs=gym_envs, inputs=inputs_dict)
        else:
            market_env(gym_envs=gym_envs, inputs=inputs_dict)

    end_time = time.perf_counter()
    total_time = end_time-start_time

    print('TOTAL TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h'.format(total_time, total_time/60, total_time/3600))