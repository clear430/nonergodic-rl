#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  gen_figures.py
usage:                  python scripts/gen_figures.py
python version:         3.9
torch verison:          1.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for aggregating data and generating all final summary figures 
    in the report for all reinforcement learning experiments.

Instructions: 
    1. Select additive environment aggregation inputs. Enter into the dictionary the common training
       hyperparameters and then into the lists the features that are varied.
    2. Select multiplicative environment aggregation inputs. Enter into each of the dictionaries the 
       common training hyperparameters and into the lists the appropriate environments.
    3. Run python file and all figures will be placed into the ./docs/figures directory.
"""

import sys
sys.path.append("./")

import os
from typing import List

from main import gym_envs
import extras.plots_figures as plots
import extras.utils as utils

# ADDITIVE ENVIRONMENTS 

# must select exactly four environments and two algorithms (can repeat)
add_envs: List[int] = [6, 7, 8, 10]    # environment ENV_KEYS from main.gym_envs dictionary
add_name: List[str] = ['Hopper', 'Walker', 'Cheetah', 'Humanoid']    # title of plotted environment results
add_algos: List[str] = ['SAC', 'TD3']
add_loss: List[str] = ['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'MSE2', 'MSE4', 'MSE6']
add_multi: List[int] = [1, 3, 5, 7, 9]

add_inputs: dict = {
    'n_trials': 10,
    'n_cumsteps': 3e5,    # 300,000 training steps
    'eval_freq': 1e3,
    'n_eval': 1e1,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    's_dist': 'N',
    'multi_steps': 1,    # default for varying critic loss functions
    }

# MULTIPLICATVE ENVIRONMENTS

# must select exactly three (non-unique) numbers of simultaneous identical gambles
n_gambles: List[int] = [1, 2, 10]

# must select exactly three (non-unique) environments for each list
# assets following the coin flip
coin_keys: List[int] = [14, 15, 16]
# assets following the dice roll
dice_keys: List[int] = [17, 18, 19]
# assets following gbm
gbm_keys: List[int] = [20, 21, 22]
# assets following gbm (discrete)
gbm_d_keys: List[int] = [23, 24, 25]

mul_inputs_td3: dict = {
    'n_trials': 5,
    'n_cumsteps': 5e4,    # 50,000 training steps
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    's_dist': 'N',
    'algo': 'TD3',
    'loss_fn': 'MSE',
    'multi_steps': 1,
    }

mul_inputs_sac: dict = {
    'n_trials': 10,
    'n_cumsteps': 5e4,    # 50,000 training steps
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    's_dist': 'N',
    'algo': 'SAC',
    'loss_fn': 'MSE',
    'multi_steps': 1,
    }

# INSURANCE SAFE HAVEN ENVIRONMENTS

# must select exactly two (non-unique) environments for each list
# single asset following the dice roll with safe haven
dice_sh_keys: List[int] = [26, 27]
dice_sh_a_keys: List[int] = [17, 28]
dice_sh_b_keys: List[int] = [18, 29]
dice_sh_c_keys: List[int] = [19, 30]

ins_inputs_td3: dict = {
    'n_trials': 10,
    'n_cumsteps': 5e3,    # 50,000 training steps
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    's_dist': 'N',
    'algo': 'TD3',
    'loss_fn': 'MSE',
    'multi_steps': 1,
    }

ins_inputs_sac: dict = {
    'n_trials': 10,
    'n_cumsteps': 5e4,    # 50,000 training steps
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    's_dist': 'N',
    'algo': 'SAC',
    'loss_fn': 'MSE',
    'multi_steps': 1,
    }

if __name__ == '__main__':

    path = './docs/figs/'    # directory to save figures

    if not os.path.exists(path):
        os.makedirs(path)

    # ADDITIVE ENVIRONMENTS

    # critic loss function plots
    plots.loss_fn_plot(path+'critic_loss.png')

    # critic loss functions
    path_loss: str = 'add_loss'
    loss_data = utils.add_loss_aggregate(add_envs, gym_envs, add_inputs, add_algos, add_loss)
    r, l, cs, ck, lt, t, sh, c, k = utils.add_summary(add_inputs, loss_data)
    plots.plot_add(add_inputs, add_name, add_loss, False, r, l, cs, ck, t, sh, k, path+path_loss)
    plots.plot_add_temp(add_inputs, add_name, add_loss, False, lt, path+path_loss)

    # multi-step returns
    path_multi: str = 'add_multi'
    multi_data = utils.add_multi_aggregate(add_envs, gym_envs, add_inputs, add_algos, add_multi)
    r, l, cs, ck, lt, t, sh, c, k = utils.add_summary(add_inputs, multi_data)
    plots.plot_add(add_inputs, add_name, add_multi, True, r, l, cs, ck, t, sh, k, path+path_multi)
    plots.plot_add_temp(add_inputs, add_name, add_multi, True, lt, path+path_multi)

    # MULTIPLICATIVE ENVIRONMENTS
    
    # coin flip
    path_env: str = 'mul_coin_inv'
    for mul_inputs in [mul_inputs_td3, mul_inputs_sac]:

        coin_inv_n1 = utils.mul_inv_aggregate(coin_keys, n_gambles[0], gym_envs, mul_inputs, safe_haven=False)
        coin_inv_n2 = utils.mul_inv_aggregate(coin_keys, n_gambles[1], gym_envs, mul_inputs, safe_haven=False)
        coin_inv_n10 = utils.mul_inv_aggregate(coin_keys, n_gambles[2], gym_envs, mul_inputs, safe_haven=False)
        r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n1)
        r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n2)
        r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n10)

        algo = '_td3' if mul_inputs == mul_inputs_td3 else '_sac'

        plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+algo+'_n1'+'.png')
        plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+algo+'_n2'+'.png')
        plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+algo+'_n10'+'.png')
        plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                                    path+path_env+algo+'_perf.png', T=1, V_0=1)
        plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                                    path+path_env+algo+'_train.png')

    # dice roll
    path_env: str = 'mul_dice_inv'
    for mul_inputs in [mul_inputs_td3, mul_inputs_sac]:

        dice_inv_n1 = utils.mul_inv_aggregate(dice_keys, n_gambles[0], gym_envs, mul_inputs, safe_haven=False)
        dice_inv_n2 = utils.mul_inv_aggregate(dice_keys, n_gambles[1], gym_envs, mul_inputs, safe_haven=False)
        dice_inv_n10 = utils.mul_inv_aggregate(dice_keys, n_gambles[2], gym_envs, mul_inputs, safe_haven=False)
        r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n1)
        r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n2)
        r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n10)

    algo = '_td3' if mul_inputs == mul_inputs_td3 else '_sac'

    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+algo+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+algo+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+algo+'_n10'+'.png')
    plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_env+algo+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_env+algo+'_train.png')

    # maximum leverage with GBM plot
    plots.plot_gbm_max_lev(path+'gbm_max_lev.png')

    # GBM
    path_env: str = 'mul_gbm_inv'
    for mul_inputs in [mul_inputs_td3, mul_inputs_sac]:

        gbm_inv_n1 = utils.mul_inv_aggregate(gbm_keys, n_gambles[0], gym_envs, mul_inputs, safe_haven=False)
        gbm_inv_n2 = utils.mul_inv_aggregate(gbm_keys, n_gambles[1], gym_envs, mul_inputs, safe_haven=False)
        gbm_inv_n10 = utils.mul_inv_aggregate(gbm_keys, n_gambles[2], gym_envs, mul_inputs, safe_haven=False)
        r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1  = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n1)
        r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n2)
        r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n10)

        algo = '_td3' if mul_inputs == mul_inputs_td3 else '_sac'

        plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+algo+'_n1'+'.png')
        plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+algo+'_n2'+'.png')
        plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+algo+'_n10'+'.png')
        plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                                path+path_env+algo+'_perf.png', T=1, V_0=1)
        plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                                path+path_env+algo+'_train.png')

    # GBM with discrete portfolio compounding 
    path_env: str = 'mul_gbm_d_inv'
    for mul_inputs in [mul_inputs_td3, mul_inputs_sac]:

        gbm_d_inv_n1 = utils.mul_inv_aggregate(gbm_d_keys, n_gambles[0], gym_envs, mul_inputs, safe_haven=False)
        gbm_d_inv_n2 = utils.mul_inv_aggregate(gbm_d_keys, n_gambles[1], gym_envs, mul_inputs, safe_haven=False)
        gbm_d_inv_n10 = utils.mul_inv_aggregate(gbm_d_keys, n_gambles[2], gym_envs, mul_inputs, safe_haven=False)
        r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1  = utils.mul_inv_n_summary(mul_inputs, gbm_d_inv_n1)
        r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, gbm_d_inv_n2)
        r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, gbm_d_inv_n10)

        algo = '_td3' if mul_inputs == mul_inputs_td3 else '_sac'

        plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+algo+'_n1'+'.png')
        plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+algo+'_n2'+'.png')
        plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+algo+'_n10'+'.png')
        plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                                path+path_env+algo+'_perf.png', T=1, V_0=1)
        plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                                path+path_env+algo+'_train.png')

    # INSURANCE SAFE HAVEN ENVIRONMENTS

    # dice roll with insurance safe haven
    path_env: str = 'mul_dice_sh'
    for ins_inputs in [ins_inputs_td3, ins_inputs_sac]:

        dice_sh = utils.mul_inv_aggregate(dice_sh_keys, 1, gym_envs, ins_inputs, safe_haven=True)
        dice_inv_a = utils.mul_inv_aggregate(dice_sh_a_keys, 1, gym_envs, ins_inputs, safe_haven=True)
        dice_inv_b = utils.mul_inv_aggregate(dice_sh_b_keys, 1, gym_envs, ins_inputs, safe_haven=True)
        dice_inv_c = utils.mul_inv_aggregate(dice_sh_c_keys, 1, gym_envs, ins_inputs, safe_haven=True)
        r_sh, le_sh, s_sh, re_sh, lo_sh, t_sh, sh_sh, c_sh, k_sh, ls_sh = utils.mul_inv_n_summary(ins_inputs, dice_sh)
        r_a, le_a, s_a, re_a, lo_a, t_a, sh_a, c_a, k_a, ls_a = utils.mul_inv_n_summary(ins_inputs, dice_inv_a, safe_haven=True)
        r_b, le_b, s_b, re_b, lo_b, t_b, sh_b, c_b, k_b, ls_b = utils.mul_inv_n_summary(ins_inputs, dice_inv_b, safe_haven=True)
        r_c, le_c, s_c, re_c, lo_c, t_c, sh_c, c_c, k_c, ls_c = utils.mul_inv_n_summary(ins_inputs, dice_inv_c, safe_haven=True)

        algo = '_td3' if ins_inputs == ins_inputs_td3 else '_sac'

        plots.plot_safe_haven(ins_inputs, r_sh, le_sh, s_sh, re_sh, lo_sh, t_sh, sh_sh, c_sh, k_sh, ls_sh, path+path_env+'.png', inv='a')
        plots.plot_safe_haven(ins_inputs, r_a, le_a, s_a, re_a, lo_a, t_a, sh_a, c_a, k_a, ls_a, path+path_env+algo+'_a.png', inv='a')
        plots.plot_safe_haven(ins_inputs, r_b, le_b, s_b, re_b, lo_b, t_b, sh_b, c_b, k_b, ls_b, path+path_env+algo+'_b.png', inv='b')
        plots.plot_safe_haven(ins_inputs, r_c, le_c, s_c, re_c, lo_c, t_c, sh_c, c_c, k_c, ls_c, path+path_env+algo+'_c.png', inv='c')
        plots.plot_inv_sh_perf(ins_inputs, r_a, le_a, s_a, re_a, ls_a, r_b, le_b, s_b, re_b, ls_b, r_c, le_c, s_c, re_c, ls_c, 
                            path+path_env+algo+'_perf.png', T=1, V_0=1)
        plots.plot_inv_sh_train(ins_inputs, lo_a, t_a, sh_a, k_a, lo_b, t_b, sh_b, k_b, lo_c, t_c, sh_c, k_c,
                                path+path_env+algo+'_train.png')
