#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  rl_tests.py
python version:         3.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for conducting tests for all reinforcement learning experiments.
"""

import sys
sys.path.append("./")

import numpy as np
import os
from typing import Dict, List, NoReturn

from algos.algo_sac import Agent_sac
from algos.networks_sac import ActorNetwork as ActorNetwork_sac
from algos.networks_sac import CriticNetwork as CriticNetwork_sac
from algos.algo_td3 import Agent_td3
from algos.networks_td3 import ActorNetwork as ActorNetwork_td3
from algos.networks_td3 import CriticNetwork as CriticNetwork_td3
from extras.replay import ReplayBuffer
from extras.replay_torch import ReplayBufferTorch

# default assertion errors
td: str = 'variable must be of type dict'
tf: str = 'variable must be of type float'
tfi: str = 'variable must be of type float for int'
ti: str = 'variable must be of type int'
tl: str = 'variable must be of type list'
ts: str = 'variable must be of type str'
gte0: str = 'quantity must be greater than or equal to 0'
gt0: str = 'quantity must be greater than 0'
gte1: str = 'quantity must be greater than or equal to 1'

def algo_tests(inputs_dict: dict) -> NoReturn:
    """
    Conduct tests on all agent algorithm and training parameters.

    Parameters:
        inputs_dict: all training and evaluation details 
    """
    # additive environment execution tests
    assert isinstance(inputs_dict['n_trials_add'], (float, int)), tfi
    assert int(inputs_dict['n_trials_add']) >= 1, gte1
    assert isinstance(inputs_dict['n_cumsteps_add'], (float, int)), tfi
    assert set(list(str(inputs_dict['n_cumsteps_add'])[2:])).issubset(set(['0', '.'])), \
        'must consist of only 2 leading non-zero digits'
    assert int(inputs_dict['n_cumsteps_add']) >= 1, gte1
    assert isinstance(inputs_dict['eval_freq_add'], (float, int)), tfi
    assert int(inputs_dict['eval_freq_add']) >= 1, gte1
    assert int(inputs_dict['eval_freq_add']) <= int(inputs_dict['n_cumsteps_add']), \
        'must be less than or equal to n_cumsteps_add'
    assert isinstance(inputs_dict['n_eval_add'], (float, int)), tfi
    assert int(inputs_dict['n_eval_add']) >= 1, gte1
    assert isinstance(inputs_dict['max_eval_reward'], (float, int)), tfi
    assert inputs_dict['max_eval_reward'] > 0, gt0

    # multiplicative environment execution tests
    assert isinstance(inputs_dict['n_trials_add'], (float, int)), tfi
    assert int(inputs_dict['n_trials_add']) >= 1, gte1
    assert isinstance(inputs_dict['n_cumsteps_mul'], (float, int)), tfi
    assert set(list(str(inputs_dict['n_cumsteps_mul'])[2:])).issubset(set(['0', '.'])), \
        'must consist of only 2 leading non-zero digits'
    assert int(inputs_dict['n_cumsteps_mul']) >= 1, gte1
    assert isinstance(inputs_dict['eval_freq_mul'], (float, int)), tfi
    assert int(inputs_dict['eval_freq_mul']) >= 1, gte1
    assert int(inputs_dict['eval_freq_mul']) <= int(inputs_dict['n_cumsteps_mul']), \
        'must be less than or equal to n_cumsteps_mul'
    assert isinstance(inputs_dict['n_eval_mul'], (float, int)), tfi
    assert int(inputs_dict['n_eval_mul']) >= 1, gte1
    assert isinstance(inputs_dict['max_eval_steps'], (float, int)), tfi
    assert int(inputs_dict['max_eval_steps']) >= 1, gte1

    # market environment execution tests
    assert isinstance(inputs_dict['n_trials_mar'], (float, int)), tfi
    assert int(inputs_dict['n_trials_mar']) >= 1, gte1
    assert isinstance(inputs_dict['train_years'], (float, int)), tfi
    assert int(inputs_dict['train_years']) > 0, gt0
    assert isinstance(inputs_dict['n_cumsteps_mar'], (float, int)), tfi
    assert int(inputs_dict['n_cumsteps_mar']) >= 1, gte1
    assert isinstance(inputs_dict['eval_freq_mar'], (float, int)), tfi
    assert int(inputs_dict['eval_freq_mar']) >= 1 
    assert isinstance(inputs_dict['n_eval_mar'], (float, int)), tfi
    assert int(inputs_dict['n_eval_mar']) >= 1, gte1
    assert isinstance(inputs_dict['test_years'], (float, int)), tfi
    assert int(inputs_dict['test_years']) > 0, gt0
    assert isinstance(inputs_dict['train_shuffle_days'], int), ti
    assert int(inputs_dict['train_shuffle_days']) >= 1, gte1
    assert isinstance(inputs_dict['test_shuffle_days'], int), ti
    assert int(inputs_dict['test_shuffle_days']) >= 1, gte1
    assert inputs_dict['train_shuffle_days'] <= int(inputs_dict['train_years'] * 252), \
        'must be less than or equal to train_years'
    assert inputs_dict['test_shuffle_days'] <= int(inputs_dict['test_years'] * 252), \
        'must be less than or equal to test_years'
    assert isinstance(inputs_dict['gap_days_min'], (int)), ti
    assert int(inputs_dict['gap_days_min']) >= 0, gte0
    assert isinstance(inputs_dict['gap_days_max'], (int)), ti
    assert int(inputs_dict['gap_days_max']) >= 0, gte0
    assert int(inputs_dict['gap_days_min']) <= int(inputs_dict['gap_days_max']), \
        'must be greater than or equal to gap_days_min'

    # learning varaible tests
    assert isinstance(inputs_dict['buffer'], (float, int)), tfi
    assert set(list(str(inputs_dict['buffer'])[2:])).issubset(set(['0', '.'])), \
        'must consist of only 2 leading non-zero digits'
    assert int(inputs_dict['buffer']) >= 1, gte1
    assert inputs_dict['buffer'] >= inputs_dict['n_cumsteps_add'], \
        'must be greater than or equal to n_cumsteps_add'
    assert inputs_dict['buffer'] >= inputs_dict['n_cumsteps_mul'], \
        'must be greater than or equal to n_cumsteps_mul'
    assert inputs_dict['buffer'] >= int(inputs_dict['n_cumsteps_mar'] * inputs_dict['train_years'] * 252), \
        'must be greater than or equal training steps'
    assert isinstance(inputs_dict['buffer_gpu'], bool), \
        'must be either True (1) or False (0)'
    assert inputs_dict['discount'] >= 0 and inputs_dict['discount'] < 1, \
        'must be within [0, 1) interval'
    assert isinstance(inputs_dict['trail'], (float, int)), tfi 
    assert int(inputs_dict['trail']) >= 1, gte1
    assert isinstance(inputs_dict['cauchy_scale'], (float, int)), tfi
    assert inputs_dict['cauchy_scale'] > 0, gt0
    assert isinstance(inputs_dict['actor_percentile'], (float, int)), tfi
    assert inputs_dict['actor_percentile'] > 0 and inputs_dict['actor_percentile'] <= 1, \
        'must be within (0, 1] interval'
    assert isinstance(inputs_dict['r_abs_zero'], (float, int))or inputs_dict['r_abs_zero'] == None, \
        'must be either real number or None'
    assert isinstance(inputs_dict['continue'], bool), \
        'must be either True (1) or False (0)'

    # critic loss aggregation tests
    assert inputs_dict['critic_mean_type'] == 'E' or 'S', \
        'must be either "E" or "S"'
    assert isinstance(inputs_dict['shadow_low_mul'], (float, int)), tfi
    assert inputs_dict['shadow_low_mul'] >= 0, gte0
    assert isinstance(inputs_dict['shadow_high_mul'], (float, int)), tfi
    assert inputs_dict['shadow_high_mul'] > 0, gt0

    # SAC hyperparameter tests
    assert isinstance(inputs_dict['sac_actor_learn_rate'], (float, int)), tfi
    assert inputs_dict['sac_actor_learn_rate'] > 0, gt0
    assert isinstance(inputs_dict['sac_critic_learn_rate'], (float, int)), tfi
    assert inputs_dict['sac_critic_learn_rate'] > 0, gt0
    assert isinstance(inputs_dict['sac_temp_learn_rate'], (float, int)), tfi
    assert inputs_dict['sac_temp_learn_rate'] > 0, gt0
    assert isinstance(inputs_dict['sac_layer_1_units'], (float, int)), tfi
    assert int(inputs_dict['sac_layer_1_units']) >= 1, gte1
    assert isinstance(inputs_dict['sac_layer_2_units'], (float, int)), tfi
    assert int(inputs_dict['sac_layer_2_units']) >= 1, gte1
    assert isinstance(inputs_dict['sac_actor_step_update'], (float, int)), tfi
    assert int(inputs_dict['sac_actor_step_update']) >= 1, gte1
    assert isinstance(inputs_dict['sac_temp_step_update'], (float, int)), tfi
    assert int(inputs_dict['sac_temp_step_update']) >= 1, gte1
    assert isinstance(inputs_dict['sac_target_critic_update'], (float, int)), tfi
    assert int(inputs_dict['sac_target_critic_update']) >= 1, gte1
    assert isinstance(inputs_dict['initial_logtemp'], (float, int)), tfi
    assert isinstance(inputs_dict['reparam_noise'], float) , tf
    assert inputs_dict['reparam_noise'] > 1e-7 and inputs_dict['reparam_noise'] < 1e-5, \
        'must be a real number in the vicinity of 1e-6'

    # TD3 hyperparameter tests
    assert isinstance(inputs_dict['td3_actor_learn_rate'], (float, int)), tfi
    assert inputs_dict['td3_actor_learn_rate'] > 0, gt0
    assert isinstance(inputs_dict['td3_critic_learn_rate'], (float, int)), tfi
    assert inputs_dict['td3_critic_learn_rate'] > 0, gt0
    assert isinstance(inputs_dict['td3_layer_1_units'], (float, int)), tfi
    assert int(inputs_dict['td3_layer_1_units']) >= 1, gte1
    assert isinstance(inputs_dict['td3_layer_2_units'], (float, int)), tfi
    assert int(inputs_dict['td3_layer_2_units']) >= 1, gte1
    assert isinstance(inputs_dict['td3_actor_step_update'], (float, int)), tfi
    assert int(inputs_dict['td3_actor_step_update']) >= 1, gte1
    assert isinstance(inputs_dict['td3_target_actor_update'], (float, int)), tfi
    assert int(inputs_dict['td3_target_actor_update']) >= 1, gte1
    assert isinstance(inputs_dict['td3_target_critic_update'], (float, int)), tfi
    assert int(inputs_dict['td3_target_critic_update']) >= 1, gte1
    assert isinstance(inputs_dict['td3_target_critic_update'], (float, int)), tfi
    assert int(inputs_dict['td3_target_critic_update']) >= 1, gte1
    assert isinstance(inputs_dict['policy_noise'], (float, int)), tfi
    assert inputs_dict['policy_noise'] >= 0, gte0
    assert isinstance(inputs_dict['target_policy_noise'], (float, int)), tfi
    assert inputs_dict['target_policy_noise'] >= 0, gte0
    assert isinstance(inputs_dict['target_policy_clip'], (float, int)), tfi
    assert inputs_dict['target_policy_clip'] >= 0, gte0

    # shared algorithm training parameter tests
    assert isinstance(inputs_dict['target_update_rate'], (float, int)), tfi
    assert inputs_dict['target_update_rate'] > 0, gt0
    assert inputs_dict['s_dist'] == ('N' or 'L') or (inputs_dict['algo_name'][0] == 'SAC' and inputs_dict['s_dist'] == 'MVN'), \
        'must be either "N" (normal=Gaussian), "L (exponential=Laplace)" and for SAC only "MVN" (multi-variate normal)'
    assert isinstance(inputs_dict['batch_size'], dict), td
    assert isinstance(inputs_dict['batch_size']['TD3'], (float, int)), tfi
    assert int(inputs_dict['batch_size']['TD3']) >= 1, gte1
    assert isinstance(inputs_dict['batch_size']['SAC'], (float, int)), tfi
    assert int(inputs_dict['batch_size']['SAC']) >= 1, gte1
    assert isinstance(inputs_dict['grad_step'], dict), td
    assert isinstance(inputs_dict['grad_step']['TD3'], (float, int)), tfi
    assert int(inputs_dict['grad_step']['TD3']) >= 1, gte1
    assert isinstance(inputs_dict['grad_step']['SAC'], (float, int)), tfi
    assert int(inputs_dict['grad_step']['SAC']) >= 1, gte1

    # input tests
    assert isinstance(inputs_dict['algo_name'], list), tl
    assert set(inputs_dict['algo_name']).issubset(set(['SAC', 'TD3'])), \
        'algorithms must be a list containing only "SAC" and/or "TD3"'
    assert isinstance(inputs_dict['critic_loss'], list), tl
    assert set(inputs_dict['critic_loss']).issubset(set(['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'CIM', 'MSE2', 'MSE4', 'MSE6'])), \
        'critic losses must be a list containing "MSE", "HUB", "MAE", "HSC", "CAU", "TCAU", "CIM", "MSE2", "MSE4", and/or "MSE6"'
    assert isinstance(inputs_dict['bootstraps'], list), tl
    assert all(isinstance(mstep, int) for mstep in inputs_dict['bootstraps']), ti
    assert all(mstep >= 1 for mstep in inputs_dict['bootstraps']), \
        'multi-steps must be a list of positive integers'
    assert isinstance(inputs_dict['past_days'], list), tl
    assert all(isinstance(days, int) for days in inputs_dict['past_days']), ti
    assert all(days >= 1 for days in inputs_dict['past_days']), \
        'observed days must be a list of positive integers'

def env_tests(gym_envs: Dict[str, list], inputs_dict: dict) -> NoReturn:
    """
    Conduct tests on details of all selected environments.

    Parameters:
        gym_envs: all environment details
        inputs_dict: all training and evaluation details 
    """
    # environment tests
    keys: List[int] = [int(env) for env in gym_envs]

    assert isinstance(inputs_dict['envs'], list), tl
    assert set(inputs_dict['envs']).issubset(set(keys)), \
        'environments must be selected from gym_envs dict keys'

    for key in inputs_dict['envs']:

        assert isinstance(gym_envs[str(key)], list), tl
        assert isinstance(gym_envs[str(key)][0], str), ts
        assert all(isinstance(x, (float, int))for x in gym_envs[str(key)][1:]), \
            'environment {} details must be a list of the form [string, real, real, real]'.format(key)
        assert int(gym_envs[str(key)][1]) >= 1, \
            'environment {} must have at least one state'.format(key)
        assert int(gym_envs[str(key)][2]) >= 1, \
            'environment {} must have at least one action'.format(key)

        if key <= 13:
            assert int(gym_envs[str(key)][3]) >= 0, gte0
            assert int(gym_envs[str(key)][3]) <= int(inputs_dict['n_cumsteps_add']), \
                'environment {} warm-up must be less than or equal to total training steps'.format(key)

        elif key <= 57:
            assert int(gym_envs[str(key)][3]) >= 0, gte0
            assert int(gym_envs[str(key)][3]) <= int(inputs_dict['n_cumsteps_mul']), \
                'environment {} warm-up must be less than or equal to total training steps'.format(key)

        else:
            assert int(gym_envs[str(key)][3]) >= 0, gte0
            assert int(gym_envs[str(key)][3]) <= int(inputs_dict['n_cumsteps_mar'] * inputs_dict['train_years'] * 252), \
                'environment {} warm-up must be less than or equal to total training steps'.format(key)

    # market environment data checks
    if any(key > 57 for key in inputs_dict['envs']):
        for key in inputs_dict['envs']:
            if key > 57:

                if inputs_dict['ENV_KEY'] <= 60:
                    assert os.path.isfile('./docs/market_data/stooq_snp.npy'), \
                        'stooq_snp.npy not generated or found in ./docs/market_data/'
                    data = np.load('./docs/market_data/stooq_snp.npy')

                elif key <= 63:
                    assert os.path.isfile('./docs/market_data/stooq_usei.npy'), \
                        'stooq_usei.npy not generated or found in ./docs/market_data/'
                    data = np.load('./docs/market_data/stooq_usei.npy')

                elif key <= 66:
                    assert os.path.isfile('./docs/market_data/stooq_minor.npy'), \
                        'stooq_minor.npy not generated or found in ./docs/market_data/'
                    data = np.load('./docs/market_data/stooq_minor.npy')

                elif key <= 69:
                    assert os.path.isfile('./docs/market_data/stooq_medium.npy'), \
                        'stooq_medium.npy not generated or found in ./docs/market_data/'
                    data = np.load('./docs/market_data/stooq_medium.npy')

                elif key <= 72:
                    assert os.path.isfile('./docs/market_data/stooq_major.npy'), \
                        'stooq_major.npy not generated or found in ./docs/market_data/'
                    data = np.load('./docs/market_data/stooq_major.npy')

                elif key <= 75:
                    assert os.path.isfile('./docs/market_data/stooq_dji.npy'), \
                        'stooq_dji.npy not generated or found in ./docs/market_data/'
                    data = np.load('./docs/market_data/stooq_dji.npy')

                elif key <= 78:
                    assert os.path.isfile('./docs/market_data/stooq_full.npy'), \
                        'stooq_full.npy not generated or found in ./docs/market_data/'
                    data = np.load('./docs/market_data/stooq_full.npy')

                time_length = data.shape[0]

                for days in inputs_dict['past_days']:
                    train_length = int(252 * inputs_dict['train_years'])
                    test_length = int(252 * inputs_dict['test_years'])
                    gap_max = int(inputs_dict['gap_days_max'])

                    sample_length = int(train_length + test_length + gap_max + days - 1)

                    assert time_length >= sample_length, \
                        'ENV_KEY {}: total time {} period for {} days must be at least as large as sample length = {}' \
                        .format(key, time_length, days, sample_length)

def method_checks(inputs_dict: dict) -> NoReturn:
    """
    Confirm the presence of numerous class methods required for agent learning.

    Parameters:
        inputs_dict: all training and evaluation details 
    """
    # SAC algorithm method checks
    if 'SAC' in inputs_dict['algo_name']:
        assert hasattr(ActorNetwork_sac, 'stochastic_uv'), 'missing SAC univariate sampling'
        assert hasattr(ActorNetwork_sac, 'stochastic_mv_gaussian'), 'missing SAC multi-variate Gaussian sampling'
        assert hasattr(ActorNetwork_sac, 'save_checkpoint'), 'missing SAC actor saving functionality'
        assert hasattr(ActorNetwork_sac, 'load_checkpoint'), 'missing SAC actor load functionality'
        assert hasattr(CriticNetwork_sac, 'forward'), 'missing SAC critic forward propagation'
        assert hasattr(CriticNetwork_sac, 'save_checkpoint'), 'missing SAC critic saving functionality'
        assert hasattr(CriticNetwork_sac, 'load_checkpoint'), 'missing SAC critic load functionality'
        assert hasattr(Agent_sac, 'select_next_action'), 'missing SAC agent action selection'
        assert hasattr(Agent_sac, 'eval_next_action'), 'missing SAC agent evaluation action selection'
        assert hasattr(Agent_sac, 'store_transistion'), 'missing SAC transition storage functionality'
        assert hasattr(Agent_sac, 'learn'), 'missing SAC agent learning functionality'
        assert hasattr(Agent_sac, 'save_models'), 'missing SAC agent save functionality'
        assert hasattr(Agent_sac, 'load_models'), 'missing SAC agent load functionality'

    # TD3 algorithm method checks
    if 'TD3' in inputs_dict['algo_name']:
        assert hasattr(ActorNetwork_td3, 'forward'), 'missing TD3 actor forward propagation'
        assert hasattr(ActorNetwork_td3, 'save_checkpoint'), 'missing TD3 actor saving functionality'
        assert hasattr(ActorNetwork_td3, 'load_checkpoint'), 'missing TD3 actor load functionality'
        assert hasattr(CriticNetwork_td3, 'forward'), 'missing TD3 critic forward propagation'
        assert hasattr(CriticNetwork_td3, 'save_checkpoint'), 'missing TD3 critic saving functionality'
        assert hasattr(CriticNetwork_td3, 'load_checkpoint'), 'missing TD3 critic load functionality'
        assert hasattr(Agent_td3, 'select_next_action'), 'missing TD3 agent action selection'
        assert hasattr(Agent_td3, 'eval_next_action'), 'missing TD3 agent evaluation action selection'
        assert hasattr(Agent_td3, 'store_transistion'), 'missing TD3 transition storage functionality'
        assert hasattr(Agent_td3, 'learn'), 'missing TD3 agent learning functionality'
        assert hasattr(Agent_td3, 'save_models'), 'missing TD3 agent save functionality'
        assert hasattr(Agent_td3, 'load_models'), 'missing TD3 agent load functionality'

    # replay buffer method checks
    if inputs_dict['buffer_gpu'] == False:
        assert hasattr(ReplayBuffer, 'store_exp'), 'missing transition store functionality'
        assert hasattr(ReplayBuffer, 'sample_exp'), 'missing uniform transition sampling functionality'
    else:
        assert hasattr(ReplayBufferTorch, 'store_exp'), 'missing transition store functionality'
        assert hasattr(ReplayBufferTorch, 'sample_exp'), 'missing uniform transition sampling functionality'