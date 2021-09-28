import sys
sys.path.append("./")

from main import gym_envs
import os
import plots_figures as plots
import utils

path = './docs/figs/'    # directory to save figures

## ADDITIVE ENVIRONMENTS 

add_inputs = {
    # execution parameters
    'n_trials': 10,
    'n_cumsteps': 3e5,
    'eval_freq': 1e3,
    'n_eval': 1e1,
    'buffer': 1e6,
    'shadow_low_mul': 1e0
    }

add_envs = [6, 7, 8, 10]
add_algos = ['SAC', 'TD3']
add_loss1 = ['MSE', 'Huber', 'MAE', 'HSC', 'Cauchy']
add_loss2 = ['MSE', 'MSE2', 'MSE4']
add_multi = [1, 3, 5, 7, 9]

## MULTIPLICATVE ENVIRONMENTS

mul_inputs = {
    # execution parameters
    'n_trials': 10,
    'n_cumsteps': 4e4,
    'eval_freq': 1e3,
    'n_eval': 1e3,
    'buffer': 1e6,
    'shadow_low_mul': 1e0,
    }

# assets following the equally likely +50%/-40% gamble
coin_n1_keys = [14, 17, 20]
coin_n2_keys = [15, 18, 21]
coin_n10_keys = [16, 19, 22]
# assets following the dice roll
dice_n1_keys = [23, 26, 29]
dice_n2_keys = [24, 27, 30]
dice_n10_keys = [25, 28, 31]
# assets following the dice roll
gbm_n1_keys = [32, 35, 38]
gbm_n2_keys = [33, 36, 39]
gbm_n10_keys = [34, 37, 40]
# assets following the dice roll with safe haven
dice_sh_keys = [41, 42]
dice_sh_a_keys = [43, 44]
dice_sh_b_keys = [45, 46]
dice_sh_c_keys = [47, 48]
# market performance across actual assets
market_keys = [49, 50, 51]

if __name__ == '__main__':

    if not os.path.exists('./docs/figs'):
        os.makedirs('./docs/figs')

    ## MULTIPLICATVE ENVIRONMENTS

    path_coin = 'mul_coin_inv'
    coin_inv_n1 = utils.mul_inv_aggregate(coin_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    coin_inv_n2 = utils.mul_inv_aggregate(coin_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    coin_inv_n10 = utils.mul_inv_aggregate(coin_n10_keys, gym_envs, mul_inputs, safe_haven=False)

    r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, coin_inv_n10)

    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_coin+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_coin+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_coin+'_n10'+'.png')

    plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_coin+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_coin+'_train.png')

    path_dice = 'mul_dice_inv'
    dice_inv_n1 = utils.mul_inv_aggregate(dice_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    dice_inv_n2 = utils.mul_inv_aggregate(dice_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    dice_inv_n10 = utils.mul_inv_aggregate(dice_n10_keys, gym_envs, mul_inputs, safe_haven=False)

    r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, dice_inv_n10)

    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_dice+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_dice+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_dice+'_n10'+'.png')

    plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_dice+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_dice+'_train.png')

    path_gbm = 'mul_gbm_inv'
    gbm_inv_n1 = utils.mul_inv_aggregate(gbm_n1_keys, gym_envs, mul_inputs, safe_haven=False)
    gbm_inv_n2 = utils.mul_inv_aggregate(gbm_n2_keys, gym_envs, mul_inputs, safe_haven=False)
    gbm_inv_n10 = utils.mul_inv_aggregate(gbm_n10_keys, gym_envs, mul_inputs, safe_haven=False)

    r1, le1, s1, re1, lo1, t1, sh1, c1, k1, ls1  = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n1)
    r2, le2, s2, re2, lo2, t2, sh2, c2, k2, ls2 = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n2)
    r10, le10, s10, re10, lo10, t10, sh10, c10, k10, ls10 = utils.mul_inv_n_summary(mul_inputs, gbm_inv_n10)

    plots.plot_inv(mul_inputs, r1, le1, s1, re1, lo1, t1, sh1, c1, k1, path+path_gbm+'_n1'+'.png')
    plots.plot_inv(mul_inputs, r2, le2, s2, re2, lo2, t2, sh2, c2, k2, path+path_gbm+'_n2'+'.png')
    plots.plot_inv(mul_inputs, r10, le10, s10, re10, lo10, t10, sh10, c10, k10, path+path_gbm+'_n10'+'.png')

    plots.plot_inv_all_n_perf(mul_inputs, r1, le1, s1, re1, r2, le2, s2, re2, r10, le10, s10, re10,
                              path+path_gbm+'_perf.png', T=1, V_0=1)
    plots.plot_inv_all_n_train(mul_inputs, lo1, t1, sh1, k1, lo2, t2, sh2, k2, lo10, t10, sh10, k10,
                              path+path_gbm+'_train.png')

    path_dice = 'mul_dice_sh'
    dice_sh = utils.mul_inv_aggregate(dice_sh_keys, gym_envs, mul_inputs, safe_haven=True)
    dice_inv_a = utils.mul_inv_aggregate(dice_sh_a_keys, gym_envs, mul_inputs, safe_haven=True)
    dice_inv_b = utils.mul_inv_aggregate(dice_sh_b_keys, gym_envs, mul_inputs, safe_haven=True)
    dice_inv_c = utils.mul_inv_aggregate(dice_sh_c_keys, gym_envs, mul_inputs, safe_haven=True)

    r_sh, le_sh, s_sh, re_sh, lo_sh, t_sh, sh_sh, c_sh, k_sh, ls_sh = utils.mul_inv_n_summary(mul_inputs, dice_sh)
    r_a, le_a, s_a, re_a, lo_a, t_a, sh_a, c_a, k_a, ls_a = utils.mul_inv_n_summary(mul_inputs, dice_inv_a, safe_haven=True)
    r_b, le_b, s_b, re_b, lo_b, t_b, sh_b, c_b, k_b, ls_b = utils.mul_inv_n_summary(mul_inputs, dice_inv_b, safe_haven=True)
    r_c, le_c, s_c, re_c, lo_c, t_c, sh_c, c_c, k_c, ls_c = utils.mul_inv_n_summary(mul_inputs, dice_inv_c, safe_haven=True)

    plots.plot_safe_haven(mul_inputs, r_sh, le_sh, s_sh, re_sh, lo_sh, t_sh, sh_sh, c_sh, k_sh, ls_sh, path+path_dice+'.png', inv='a')
    plots.plot_safe_haven(mul_inputs, r_a, le_a, s_a, re_a, lo_a, t_a, sh_a, c_a, k_a, ls_a, path+path_dice+'_a.png', inv='a')
    plots.plot_safe_haven(mul_inputs, r_b, le_b, s_b, re_b, lo_b, t_b, sh_b, c_b, k_b, ls_b, path+path_dice+'_b.png', inv='b')
    plots.plot_safe_haven(mul_inputs, r_c, le_c, s_c, re_c, lo_c, t_c, sh_c, c_c, k_c, ls_c, path+path_dice+'_c.png', inv='c')

    plots.plot_inv_sh_perf(mul_inputs, r_a, le_a, s_a, re_a, ls_a, r_b, le_b, s_b, re_b, ls_b, r_c, le_c, s_c, re_c, ls_c, 
                           path+path_dice+'_perf.png', T=1, V_0=1)
    plots.plot_inv_sh_train(mul_inputs, lo_a, t_a, sh_a, k_a, lo_b, t_b, sh_b, k_b, lo_c, t_c, sh_c, k_c,
                            path+path_dice+'_train.png')