import sys
sys.path.append("./")

from main import gym_envs
import numpy as np
import os
import plots_figures as plots
from typing import Tuple
import utils

add_inputs = {
    # execution parameters
    'n_trials': 10,
    'n_cumsteps': 3e5,
    'eval_freq': 1e3,
    'n_eval': 1e1,
    'buffer': 1e6,
    'shadow_low_mul': 1e0
    }

mul_inputs = {
    # execution parameters
    'n_trials': 5,
    'n_cumsteps': 2e4,
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'shadow_low_mul': 1e0,
    }
        
path = './docs/figs/'    # directory to save figures

# additive experiment details
add_envs = [6, 7, 8, 10]
add_algos = ['SAC', 'TD3']
add_loss1 = ['MSE', 'Huber', 'MAE', 'HSC', 'Cauchy']
add_loss2 = ['MSE', 'MSE2', 'MSE4']
add_multi = [1, 3, 5, 7, 9]

# multiplicative experiment details

# assets following the equally likely +50%/-40% gamble
coin_n1_keys = [14, 17, 20]
coin_n2_keys = [15, 18, 21]
coin_n10_keys = [16, 19, 22]
# assets following the dice roll
dice_n1_keys = [23, 26, 29]
dice_n2_keys = [24, 27, 30]
dice_n10_keys = [25, 28, 31]

if __name__ == '__main__':

    if not os.path.exists('./docs/figs'):
        os.makedirs('./docs/figs')

    path_coin = 'mul_coin_inv'
    coin_inv_n1 = utils.mul_inv_aggregate(coin_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    coin_inv_n2 = utils.mul_inv_aggregate(coin_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    coin_inv_n10 = utils.mul_inv_aggregate(coin_n10_keys, gym_envs, mul_inputs, safe_haven=False)

    r1, le1, s1, re1, lo1, t1, sh1, c1, he1 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, he2= utils.mul_inv_n_summary(mul_inputs, coin_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, he10 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n10)

    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, he1, path+path_coin+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, he2, path+path_coin+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, he10, path+path_coin+'_n10'+'.png')

    plots.plot_inv_all_n(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                         path+path_coin+'.png', T=1, V_0=1)

    path_dice = 'mul_dice_inv'
    dice_inv_n1 = utils.mul_inv_aggregate(dice_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    dice_inv_n2 = utils.mul_inv_aggregate(dice_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    dice_inv_n10 = utils.mul_inv_aggregate(dice_n10_keys, gym_envs, mul_inputs, safe_haven=False)

    r1, le1, s1, re1, lo1, t1, sh1, c1, he1 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, he2= utils.mul_inv_n_summary(mul_inputs, dice_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, he10 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n10)

    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, he1, path+path_dice+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, he2, path+path_dice+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, he10, path+path_dice+'_n10'+'.png')

    plots.plot_inv_all_n(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                         path+path_dice+'.png', T=1, V_0=1)