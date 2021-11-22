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
    3. Run python file and upon completion, all learned PyTorch parameters will be placed ./models, and 
       data/plots regarding training into ./results/ directories titled. These will be organised by the reward dynamic 
       (additive or multiplicative or market). Then inside each will exist directories titled by full env_id
       containing all output data and summary plots.
"""

import numpy as np
import time
from typing import Dict, List

from scripts.rl_additive import additive_env
from scripts.rl_market import market_env
from scripts.rl_multiplicative import multiplicative_env
from scripts.rl_tests import algo_tests, env_tests, method_checks

# model-free off-policy agents: list ['SAC', 'TD3']
algo_name: List[str] = ['TD3']

# critic loss functions: list ['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'CIM', 'MSE2', 'MSE4', 'MSE6']
critic_loss: List[str] = ['MSE']

# bootstrapping of target critic values and discounted rewards: list [integer > 0] 
multi_steps: List[int] = [1]

# number of previous days observed for ONLY market environments (Markov if =1): list [integer > 0]
obs_days: List[int] = [1]

# environments to train agent: list [integer ENV_KEY from gym_envs]
envs: List[int] = [58]

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

    # S&P500 index (^SPX)
    '58': ['SNP_InvA', 2, 1, 1e3],      '59': ['SNP_InvB', 2, 2, 1e3],      '60': ['SNP_InvC', 2, 3, 1e3],
    # US equity indicies (^SPX, ^NDX, ^DJIA)
    '61': ['EI_InvA', 4, 3, 1e3],       '62': ['EI_InvB', 4, 4, 1e3],       '63': ['EI_InvC', 4, 5, 1e3],
    # US-listed equity indicies and a few commoditites
    '64': ['Minor_InvA', 7, 6, 1e3],    '65': ['Minor_InvB', 7, 7, 1e3],    '66': ['Minor_InvC', 7, 8, 1e3],
    # US-listed equity indicies and several commoditites
    '67': ['Medium_InvA', 10, 9, 1e3],  '68': ['Medium_InvB', 10, 10, 1e3], '69': ['Medium_InvC', 10, 11, 1e3],
    # US-listed equity indicies and many commoditites
    '70': ['Major_InvA', 15, 14, 1e3],  '71': ['Major_InvB', 15, 15, 1e3],  '72': ['Major_InvC', 15, 16, 1e3],
    # US equity indicies and 26/30 Dow Jones (^DJIA) components
    '73': ['DJI_InvA', 30, 29, 1e3],    '74': ['DJI_InvB', 30, 30, 1e3],    '75': ['DJI_InvC', 30, 31, 1e3],
    # Combined Major + DJI market
    '76': ['Full_InvA', 41, 40, 1e3],   '77': ['Full_InvB', 41, 41, 1e3],   '78': ['Full_InvC', 41, 42, 1e3],
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

    # market environment execution parameters
    'n_trials_mar': 10,                         # ibid.
    'train_years': 4,                           # length of training periods (252 day years)
    'n_cumsteps_mar': 8e1,                      # total training periods
    'eval_freq_mar': 1,                         # training periods between evaluation episodes
    'n_eval_mar': 1e1,                          # ibid.
    'test_years': 1,                            # length of testing periods (252 day years)
    'train_shuffle_days': 10,                   # interval size to be shuffled for training
    'test_shuffle_days': 5,                     # interval size to be shuffled testing (inference)
    'gap_days_min': 5,                          # minimum spacing between training/testing windows
    'gap_days_max': 20,                         # maximum spacing between training/testing windows

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
    'past_days': obs_days,
    'envs': envs,
    'ENV_KEY': 0
    }

if __name__ == '__main__':

    # conduct tests
    algo_tests(inputs_dict)
    env_tests(gym_envs, inputs_dict)
    method_checks(inputs_dict)

    start_time = time.perf_counter()

    for env_key in envs:
        inputs_dict['ENV_KEY'] = env_key 
            
        if env_key <= 13:
            additive_env(gym_envs=gym_envs, inputs=inputs_dict)

        elif env_key <= 57:
            multiplicative_env(gym_envs=gym_envs, inputs=inputs_dict)

        else:
            if env_key <= 60:
                data = np.load('./docs/market_data/stooq_snp.npy')
            elif env_key <= 63:
                data = np.load('./docs/market_data/stooq_usei.npy')
            elif env_key <= 66:
                data = np.load('./docs/market_data/stooq_minor.npy')
            elif env_key <= 69:
                data = np.load('./docs/market_data/stooq_medium.npy')
            elif env_key <= 72:
                data = np.load('./docs/market_data/stooq_major.npy')
            elif env_key <= 75:
                data = np.load('./docs/market_data/stooq_dji.npy')
            elif env_key <= 78:
                data = np.load('./docs/market_data/stooq_full.npy')

            for days in inputs_dict['past_days']:
                market_env(gym_envs=gym_envs, inputs=inputs_dict, market_data=data, obs_days=days)

    end_time = time.perf_counter()
    total_time = end_time-start_time

    print('TOTAL TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h'.format(total_time, total_time/60, total_time/3600))