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
    1. Select additive environment aggregation inputs.
    2. Select multiplicative environment aggregation inputs.
    3. Run python file all figures data will be placed into the ./docs/figures directory.
"""

import sys
sys.path.append("./")

import os

from main import gym_envs
import extras.plots_figures as plots
import extras.utils as utils

# ADDITIVE ENVIRONMENTS 

add_inputs = {
    'n_trials': 10,
    'n_cumsteps': 3e5,
    'eval_freq': 1e3,
    'n_eval': 1e1,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    'shadow_low_mul': 1e0,
    's_dist': 'N',
    'algo': 'TD3',
    'loss_fn': 'MSE',
    'multi_steps': 1,
    }

add_envs = [6, 7, 8, 10]
add_name = ['Hopper', 'Walker', 'Cheetah', 'Humanoid']
add_algos = ['SAC', 'TD3']
add_loss = ['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'MSE2', 'MSE4', 'MSE6']
add_multi = [1, 3, 5, 7, 9]

# MULTIPLICATVE ENVIRONMENTS

mul_inputs_1 = {
    'n_trials': 10,
    'n_cumsteps': 4e4,
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    'shadow_low_mul': 1e0,
    's_dist': 'N',
    'algo': 'TD3',
    'loss_fn': 'MSE',
    'multi_steps': 1,
    }

mul_inputs_2 = {
    'n_trials': 10,
    'n_cumsteps': 8e4,
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'critic_mean_type': 'E',
    'shadow_low_mul': 1e0,
    's_dist': 'N',
    'algo': 'TD3',
    'loss_fn': 'MSE',
    'multi_steps': 1,
    }

# assets following the equally likely +50%/-40% gamble
coin_n1_keys = [14, 17, 20]
coin_n2_keys = [15, 18, 21]
coin_n10_keys = [16, 19, 22]
# assets following the dice roll
dice_n1_keys = [23, 26, 29]
dice_n2_keys = [24, 27, 30]
dice_n10_keys = [25, 28, 31]
# assets following gbm
gbm_n1_keys = [32, 35, 38]
gbm_n2_keys = [33, 36, 39]
gbm_n10_keys = [34, 37, 40]
# assets following bgm (discrete)
gbm_d_n1_keys = [41, 44, 47]
gbm_d_n2_keys = [42, 45, 48]
gbm_d_n10_keys = [43, 46, 49]
# assets following the dice roll with safe haven
dice_sh_keys = [50, 51]
dice_sh_a_keys = [52, 53]
dice_sh_b_keys = [54, 55]
dice_sh_c_keys = [56, 57]
# market performance across actual assets
snp_keys = [58, 59, 60]
market_keys = [61]

if __name__ == '__main__':

    path = './docs/figs/'    # directory to save figures

    if not os.path.exists('./docs/figs'):
        os.makedirs('./docs/figs')

    # ADDITIVE ENVIRONMENTS

    plots.loss_fn_plot(path+'critic_loss.png')

    path_loss = 'add_loss'
    loss_data = utils.add_loss_aggregate(add_envs, gym_envs, add_inputs, add_algos, add_loss)
    r, l, cs, ck, lt, t, sh, c, k = utils.add_summary(add_inputs, loss_data)
    plots.plot_add(add_inputs, add_name, add_loss, False, r, l, cs, ck, t, sh, k, path+path_loss)
    plots.plot_add_temp(add_inputs, add_name, add_loss, False, lt, path+path_loss)

    path_multi = 'add_multi'
    multi_data = utils.add_multi_aggregate(add_envs, gym_envs, add_inputs, add_algos, add_multi)
    r, l, cs, ck, lt, t, sh, c, k = utils.add_summary(add_inputs, multi_data)
    plots.plot_add(add_inputs, add_name, add_multi, True, r, l, cs, ck, t, sh, k, path+path_multi)
    plots.plot_add_temp(add_inputs, add_name, add_multi, True, lt, path+path_multi)

    # MULTIPLICATIVE ENVIRONMENTS
    
    path_env = 'mul_coin_inv'
    mul_inputs = mul_inputs_1
    coin_inv_n1 = utils.mul_inv_aggregate(coin_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    coin_inv_n2 = utils.mul_inv_aggregate(coin_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    coin_inv_n10 = utils.mul_inv_aggregate(coin_n10_keys, gym_envs, mul_inputs, safe_haven=False)
    r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n10)
    plots.plot_inv(mul_inputs_1, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+'_n1'+'.png')
    plots.plot_inv(mul_inputs_1, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+'_n2'+'.png')
    plots.plot_inv(mul_inputs_1, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+'_n10'+'.png')
    plots.plot_inv_all_n_perf(mul_inputs_1, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_env+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs_1, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_env+'_train.png')

    path_env = 'mul_dice_inv'
    mul_inputs = mul_inputs_1
    dice_inv_n1 = utils.mul_inv_aggregate(dice_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    dice_inv_n2 = utils.mul_inv_aggregate(dice_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    dice_inv_n10 = utils.mul_inv_aggregate(dice_n10_keys, gym_envs, mul_inputs, safe_haven=False)
    r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n10)
    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+'_n10'+'.png')
    plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_env+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_env+'_train.png')

    plots.plot_gbm_max_lev(path+'gbm_max_lev.png')

    path_env = 'mul_gbm_inv'
    mul_inputs = mul_inputs_2
    gbm_inv_n1 = utils.mul_inv_aggregate(gbm_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    gbm_inv_n2 = utils.mul_inv_aggregate(gbm_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    gbm_inv_n10 = utils.mul_inv_aggregate(gbm_n10_keys, gym_envs, mul_inputs, safe_haven=False)
    r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1  = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n10)
    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+'_n10'+'.png')
    plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_env+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_env+'_train.png')

    path_env = 'mul_gbm_d_inv'
    mul_inputs = mul_inputs_2
    gbm_d_inv_n1 = utils.mul_inv_aggregate(gbm_d_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    gbm_d_inv_n2 = utils.mul_inv_aggregate(gbm_d_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    gbm_d_inv_n10 = utils.mul_inv_aggregate(gbm_d_n10_keys, gym_envs, mul_inputs, safe_haven=False)
    r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1  = utils.mul_inv_n_summary(mul_inputs, gbm_d_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, gbm_d_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, gbm_d_inv_n10)
    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_env+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_env+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_env+'_n10'+'.png')
    plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_env+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_env+'_train.png')

    path_env = 'mul_dice_sh'
    mul_spitz_inputs = mul_inputs_1
    mul_inputs = mul_inputs_1
    dice_sh = utils.mul_inv_aggregate(dice_sh_keys, gym_envs, mul_spitz_inputs, safe_haven=True)
    dice_inv_a = utils.mul_inv_aggregate(dice_sh_a_keys, gym_envs, mul_inputs, safe_haven=True)
    dice_inv_b = utils.mul_inv_aggregate(dice_sh_b_keys, gym_envs, mul_inputs, safe_haven=True)
    dice_inv_c = utils.mul_inv_aggregate(dice_sh_c_keys, gym_envs, mul_inputs, safe_haven=True)
    r_sh, le_sh, s_sh, re_sh, lo_sh, t_sh, sh_sh, c_sh, k_sh, ls_sh = utils.mul_inv_n_summary(mul_spitz_inputs, dice_sh)
    r_a, le_a, s_a, re_a, lo_a, t_a, sh_a, c_a, k_a, ls_a = utils.mul_inv_n_summary(mul_inputs, dice_inv_a, safe_haven=True)
    r_b, le_b, s_b, re_b, lo_b, t_b, sh_b, c_b, k_b, ls_b = utils.mul_inv_n_summary(mul_inputs, dice_inv_b, safe_haven=True)
    r_c, le_c, s_c, re_c, lo_c, t_c, sh_c, c_c, k_c, ls_c = utils.mul_inv_n_summary(mul_inputs, dice_inv_c, safe_haven=True)
    plots.plot_safe_haven(mul_spitz_inputs, r_sh, le_sh, s_sh, re_sh, lo_sh, t_sh, sh_sh, c_sh, k_sh, ls_sh, path+path_env+'.png', inv='a')
    plots.plot_safe_haven(mul_inputs, r_a, le_a, s_a, re_a, lo_a, t_a, sh_a, c_a, k_a, ls_a, path+path_env+'_a.png', inv='a')
    plots.plot_safe_haven(mul_inputs, r_b, le_b, s_b, re_b, lo_b, t_b, sh_b, c_b, k_b, ls_b, path+path_env+'_b.png', inv='b')
    plots.plot_safe_haven(mul_inputs, r_c, le_c, s_c, re_c, lo_c, t_c, sh_c, c_c, k_c, ls_c, path+path_env+'_c.png', inv='c')
    plots.plot_inv_sh_perf(mul_inputs, r_a, le_a, s_a, re_a, ls_a, r_b, le_b, s_b, re_b, ls_b, r_c, le_c, s_c, re_c, ls_c, 
                           path+path_env+'_perf.png', T=1, V_0=1)
    plots.plot_inv_sh_train(mul_inputs, lo_a, t_a, sh_a, k_a, lo_b, t_b, sh_b, k_b, lo_c, t_c, sh_c, k_c,
                            path+path_env+'_train.png')