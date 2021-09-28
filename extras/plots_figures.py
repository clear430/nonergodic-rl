import sys
sys.path.append("./")

import extras.utils as utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch as T
from typing import List

def loss_fn_plot(filename_png: str):
    """
    Plot of critic loss functions about the origin.

    Parameters:
        filename_png (directory): save path of plot
    """
    pdf = T.distributions.normal.Normal(0, 10)
    q = int(1e6)

    a = pdf.sample((q,))
    b = pdf.sample((q,))
    c = b - a

    mse = utils.mse(a, b, 0)
    mse2 = utils.mse(a, b, 2)
    mse4 = utils.mse(a, b, 4)
    huber = utils.huber(a, b)
    mae = utils.mae(a, b)
    hsc = utils.hypersurface(a, b)
    cauchy = utils.cauchy(a, b, 1)

    size = 3
    cols = ['C'+str(x) for x in range(7)]
    l = ['MSE', 'MSE2', 'MSE4', 'Huber', 'MAE', 'HSC', 'Cauchy']

    plt.scatter(c, mse, s=size, c=cols[0])
    plt.scatter(c, mse2, s=size, c=cols[1])
    plt.scatter(c, mse4, s=size, c=cols[2])
    plt.scatter(c, huber, s=size, c=cols[3])
    plt.scatter(c, mae, s=size, c=cols[4])
    plt.scatter(c, hsc, s=size, c=cols[5])
    plt.scatter(c, cauchy, s=size, c=cols[6])    

    plt.xlim((-6, 6))
    plt.ylim((0, 5))
    plt.tick_params(axis='both', which='major', labelsize='small')
    plt.title('Loss', size='medium')
    plt.legend(l, loc='lower right', ncol=1, frameon=False, fontsize='medium', markerscale=6)
    plt.tight_layout()

    plt.savefig(filename_png, dpi=300, format='png')

def plot_critic_2d(inputs: dict, data: np.ndarray, algo_name: List[str], critic_name: List[str], filename_png: str):
    """
    2D plot of Mean, MAD, and STD of scores and twin critic loss during evaluation episodes for all trials in environment.
    
    Parameters:
        inputs: dictionary containing all execution details
        eval_log: log of episode data for all trials
        filename_png (directory): save path of plot
    """
    algos = data.shape[0]
    closs = data.shape[1]
    cum_steps_log = data[0, 0, 0, :, 0, -1]

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(inputs['n_cumsteps'] / inputs['eval_freq'])
    count_y = int(inputs['n_trials'] * int(inputs['n_eval']))

    scores = np.zeros((algos, closs, count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(inputs['max_eval_reward'])
    loss = np.zeros((algos, closs, count_x, count_y)) 

    for a in range(algos):
        for l in range(closs):
            for t in range(count_x):
                for n in range(inputs['n_trials']):
                    for s in range(int(inputs['n_eval'])):
                        scores[a, l, t, s + n * int(inputs['n_eval'])] = data[a, l, n, t, s, 1]
                        loss[a, l, t, s + n * int(inputs['n_eval'])] = np.mean(data[a, l, n, t, s, 3:5])

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    cols = ['C'+str(x) for x in range(closs)]
    name = inputs['env_id']
    name = name[0:name.index("Bullet")]

    fig, axs = plt.subplots(2, 2)
    
    for sc in range(2):
        for cl in range(closs):
        
            score_cl = scores[sc, cl]

            score_mean = np.mean(score_cl, axis=1, keepdims=True)
            score_max, score_min = np.max(score_cl, axis=1, keepdims=True), np.min(score_cl, axis=1, keepdims=True)
            score_mad = np.mean(np.abs(score_cl - score_mean), axis=1, keepdims=True)
            score_mad_up = np.minimum(score_max, score_mean+score_mad, score_limit).reshape(-1)
            score_mad_lo = np.maximum(score_min, score_mean-score_mad).reshape(-1)
            score_mean = score_mean.reshape(-1)

            axs[0, sc].plot(x_steps, score_mean, color=cols[cl], linewidth=1)
            axs[0, sc].fill_between(x_steps, score_mad_lo, score_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[0, sc].grid(True, linewidth=0.2)
            axs[0, sc].set_title(algo_name[sc])

    axs[0, 0].set_ylabel('Score') 
    
    for sc in range(2):
        for cl in range(closs):
        
            loss_cl = loss[sc, cl]

            loss_mean = np.mean(loss_cl, axis=1, keepdims=True)
            loss_max, loss_min = np.max(loss_cl, axis=1, keepdims=True), np.min(loss_cl, axis=1, keepdims=True)
            loss_mad = np.mean(np.abs(loss_cl - loss_mean), axis=1, keepdims=True)
            loss_mad_up = np.minimum(loss_max, loss_mean+loss_mad).reshape(-1)
            loss_mad_lo = np.maximum(loss_min, loss_mean-loss_mad).reshape(-1)
            loss_mean = loss_mean.reshape(-1)

            axs[1, sc].plot(x_steps, loss_mean, color=cols[cl], linewidth=1)
            axs[1, sc].fill_between(x_steps, loss_mad_lo, loss_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[1, sc].grid(True, linewidth=0.2)

    axs[1, 0].set_ylabel('Twin Critic Loss')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')

    fig.subplots_adjust(bottom=0.175)
    fig.legend(critic_name, loc='lower center', ncol=closs, frameon=False, fontsize='medium')
    fig.suptitle(name, fontsize=14)

    plt.savefig(filename_png, dpi=300, format='png')

def plot_critic_loss(inputs: dict, data: np.ndarray, algo_name: List[str], critic_name: List[str], filename_png: str):
    """
    Plot of score, critic losses Cauchy scale, and CIM kernel for two algorithms and all critic loss functions 
    for a single environments.

    Parameters:
        inputs: dictionary containing all execution details
        data: mega combined array of everything
        algo_name: name of both RL algorithms
        cirtic_name: name of all critic loss functions
        filename_png (directory): save path of plot
    """
    algos = data.shape[0]
    closs = data.shape[1]
    cum_steps_log = data[0, 0, 0, :, 0, -1]

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(inputs['n_cumsteps'] / inputs['eval_freq'])
    count_y = int(inputs['n_trials'] * int(inputs['n_eval']))
    count_z = int(inputs['n_trials'] )

    scores = np.zeros((algos, closs, count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(inputs['max_eval_reward'])
    loss = np.zeros((algos, closs, count_x, count_z * 2))
    scale = np.zeros((algos, closs, count_x, count_z * 2)) 
    kernel = np.zeros((algos, closs, count_x, count_z * 2)) 

    for a in range(algos):
        for l in range(closs):
            for t in range(count_x):
                for n in range(inputs['n_trials']):

                    loss[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 3:5]
                    scale[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 11:13]
                    kernel[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 13:15]

                    for s in range(int(inputs['n_eval'])):
                        scores[a, l, t, s + n * int(inputs['n_eval'])] = data[a, l, n, t, s, 1]

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    cols = ['C'+str(x) for x in range(closs)]
    name = inputs['env_id']
    name = name[0:name.index("Bullet")]

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8,10))
    
    for sc in range(2):
        for cl in range(closs):
        
            score_cl = scores[sc, cl]

            score_mean = np.mean(score_cl, axis=1, keepdims=True)
            score_max, score_min = np.max(score_cl, axis=1, keepdims=True), np.min(score_cl, axis=1, keepdims=True)
            score_mad = np.mean(np.abs(score_cl - score_mean), axis=1, keepdims=True)
            score_mad_up = np.minimum(score_max, score_mean+score_mad, score_limit).reshape(-1)
            score_mad_lo = np.maximum(score_min, score_mean-score_mad).reshape(-1)
            score_mean = score_mean.reshape(-1)

            axs[0, sc].plot(x_steps, score_mean, color=cols[cl], linewidth=1)
            axs[0, sc].fill_between(x_steps, score_mad_lo, score_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[0, sc].grid(True, linewidth=0.2)
            axs[0, sc].set_title(algo_name[sc])
            axs[0, sc].xaxis.set_ticklabels([])

    # axs[0, 0].set_ylim(0)
    axs[0, 0].set_ylabel('Score') 
    axs[0, 0].text(1.05, -0.2, "(a)", size='medium', transform=axs[0, 0].transAxes)
    
    for sc in range(2):
        for cl in range(closs):
        
            loss_cl = loss[sc, cl]

            loss_mean = np.mean(loss_cl, axis=1, keepdims=True)
            loss_max, loss_min = np.max(loss_cl, axis=1, keepdims=True), np.min(loss_cl, axis=1, keepdims=True)
            loss_mad = np.mean(np.abs(loss_cl - loss_mean), axis=1, keepdims=True)
            loss_mad_up = np.minimum(loss_max, loss_mean+loss_mad).reshape(-1)
            loss_mad_lo = np.maximum(loss_min, loss_mean-loss_mad).reshape(-1)
            loss_mean = loss_mean.reshape(-1)

            axs[1, sc].plot(x_steps, loss_mean, color=cols[cl], linewidth=1)
            axs[1, sc].fill_between(x_steps, loss_mad_lo, loss_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[1, sc].grid(True, linewidth=0.2)
            axs[1, sc].xaxis.set_ticklabels([])

    # axs[1, 0].set_ylim(0)
    axs[1, 0].set_ylabel('Critic Loss')
    axs[1, 0].text(1.05, -0.2, "(b)", size='medium', transform=axs[1, 0].transAxes)

    for sc in range(2):
        for cl in range(closs):
        
            scale_cl = scale[sc, cl] / 1e5

            scale_mean = np.mean(scale_cl, axis=1, keepdims=True)
            scale_max, scale_min = np.max(scale_cl, axis=1, keepdims=True), np.min(scale_cl, axis=1, keepdims=True)
            scale_mad = np.mean(np.abs(scale_cl - scale_mean), axis=1, keepdims=True)
            scale_mad_up = np.minimum(scale_max, scale_mean+scale_mad).reshape(-1)
            scale_mad_lo = np.maximum(scale_min, scale_mean-scale_mad).reshape(-1)
            scale_mean = scale_mean.reshape(-1)

            axs[2, sc].plot(x_steps, scale_mean, color=cols[cl], linewidth=1)
            axs[2, sc].fill_between(x_steps, scale_mad_lo, scale_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[2, sc].grid(True, linewidth=0.2)
            axs[2, sc].xaxis.set_ticklabels([])

    # axs[2, 0].set_ylim(0)
    axs[2, 0].set_ylabel('Cauchy Scale ω')
    axs[2, 0].text(1.05, -0.2, "(c)", size='medium', transform=axs[2, 0].transAxes)

    for sc in range(2):
        for cl in range(closs):
        
            kernel_cl = kernel[sc, cl]

            kernel_mean = np.mean(kernel_cl, axis=1, keepdims=True)
            kernel_max, kernel_min = np.max(kernel_cl, axis=1, keepdims=True), np.min(kernel_cl, axis=1, keepdims=True)
            kernel_mad = np.mean(np.abs(kernel_cl - kernel_mean), axis=1, keepdims=True)
            kernel_mad_up = np.minimum(kernel_max, kernel_mean+kernel_mad).reshape(-1)
            kernel_mad_lo = np.maximum(kernel_min, kernel_mean-kernel_mad).reshape(-1)
            kernel_mean = kernel_mean.reshape(-1)

            axs[3, sc].plot(x_steps, kernel_mean, color=cols[cl], linewidth=1)
            axs[3, sc].fill_between(x_steps, kernel_mad_lo, kernel_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[3, sc].grid(True, linewidth=0.2)

    # axs[3, 0].set_ylim(0)
    axs[3, 0].set_ylabel('CIM Kernel Size σ')
    axs[3, 0].text(1.05, -0.2, "(d)", size='medium', transform=axs[3, 0].transAxes)
    axs[3, 0].set_xlabel('Steps (1e'+str(exp)+')')

    fig.subplots_adjust(bottom=0.1, hspace=0.3)
    fig.legend(critic_name, loc='lower center', ncol=closs, frameon=False, fontsize='medium')
    # fig.suptitle(name, fontsize='xx-large', y=0.95)

    plt.savefig(filename_png, dpi=300, format='png')

def plot_critic_shadow(inputs: dict, data: np.ndarray, algo_name: List[str], critic_name: List[str], filename_png: str):
    """
    Plot of shadow losses and tail index for two algorithms and all critic loss functions for a single environments.

    Parameters:
        inputs: dictionary containing all execution details
        data: mega combined array of everything
        algo_name: name of both RL algorithms
        cirtic_name: name of all critic loss functions
        filename_png (directory): save path of plot
    """
    algos = data.shape[0]
    closs = data.shape[1]
    cum_steps_log = data[0, 0, 0, :, 0, -1]

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(inputs['n_cumsteps'] / inputs['eval_freq'])
    count_y = int(inputs['n_trials'] * int(inputs['n_eval']))
    count_z = int(inputs['n_trials'] )

    scores = np.zeros((algos, closs, count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(inputs['max_eval_reward'])
    loss = np.zeros((algos, closs, count_x, count_z * 2))
    shadow = np.zeros((algos, closs, count_x, count_z * 2)) 
    alpha = np.zeros((algos, closs, count_x, count_z * 2)) 

    for a in range(algos):
        for l in range(closs):
            for t in range(count_x):
                for n in range(inputs['n_trials']):

                    # loss[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 3:5]
                    shadow[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 5:7]
                    alpha[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 7:9]

                    # for s in range(int(inputs['n_eval'])):
                    #     scores[a, l, t, s + n * int(inputs['n_eval'])] = data[a, l, n, t, s, 1]

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    cols = ['C'+str(x) for x in range(closs)]
    name = inputs['env_id']
    name = name[0:name.index("Bullet")]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    
    for sc in range(2):
        for cl in range(closs):
        
            shadow_cl = shadow[sc, cl] / 1e5

            shadow_mean = np.mean(shadow_cl, axis=1, keepdims=True)
            shadow_max, shadow_min = np.max(shadow_cl, axis=1, keepdims=True), np.min(shadow_cl, axis=1, keepdims=True)
            shadow_mad = np.mean(np.abs(shadow_cl - shadow_mean), axis=1, keepdims=True)
            shadow_mad_up = np.minimum(shadow_max, shadow_mean+shadow_mad).reshape(-1)
            shadow_mad_lo = np.maximum(shadow_min, shadow_mean-shadow_mad).reshape(-1)
            shadow_mean = shadow_mean.reshape(-1)

            axs[0, sc].plot(x_steps, shadow_mean, color=cols[cl], linewidth=1)
            axs[0, sc].fill_between(x_steps, shadow_mad_lo, shadow_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[0, sc].grid(True, linewidth=0.2)
            axs[0, sc].set_title(algo_name[sc])
            axs[0, sc].xaxis.set_ticklabels([])

    # axs[1, 0].set_ylim(0)
    axs[0, 0].set_ylabel('Shadow Loss (1e5)')
    axs[0, 0].text(1.05, -0.2, "(a)", size='medium', transform=axs[0, 0].transAxes)

    for sc in range(2):
        for cl in range(closs):
        
            alpha_cl = alpha[sc, cl]
            alpha_cl = np.nan_to_num(alpha_cl, 1e9)

            alpha_mean = np.mean(alpha_cl, axis=1, keepdims=True)
            alpha_max, alpha_min = np.max(alpha_cl, axis=1, keepdims=True), np.min(alpha_cl, axis=1, keepdims=True)
            alpha_mad = np.mean(np.abs(alpha_cl - alpha_mean), axis=1, keepdims=True)
            alpha_mad_up = np.minimum(alpha_max, alpha_mean+alpha_mad).reshape(-1)
            alpha_mad_lo = np.maximum(alpha_min, alpha_mean-alpha_mad).reshape(-1)
            alpha_mean = alpha_mean.reshape(-1)

            axs[1, sc].plot(x_steps, alpha_mean, color=cols[cl], linewidth=1)
            axs[1, sc].fill_between(x_steps, alpha_mad_lo, alpha_mad_up, facecolor=cols[cl], alpha=0.1)
            axs[1, sc].grid(True, linewidth=0.2)

    # axs[1, 0].set_ylim(0)
    axs[1, 0].set_ylabel('Tail Index α')
    axs[1, 0].text(1.05, -0.2, "(b)", size='medium', transform=axs[1, 0].transAxes)
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')

    fig.subplots_adjust(bottom=0.15, hspace=0.2)
    fig.legend(critic_name, loc='lower center', ncol=closs, frameon=False, fontsize='medium')
    # fig.suptitle(name, fontsize='xx-large', y=0.95)

    plt.savefig(filename_png, dpi=300, format='png')

def plot_temp(inputs: dict, data: np.ndarray, env_name: List[str], critic_name: List[str], filename_png: str):
    """
    Plot SAC sutomatically tuned temperature hyperparameters for only two environments and all critic loss functions.

    Parameters:
        inputs: dictionary containing all execution details
        data: mega combined array of everything
        env_name: name of both environments
        cirtic_name: name of all critic loss functions
        filename_png (directory): save path of plot
    """
    envs = data.shape[0]
    closs = data.shape[1]
    cum_steps_log = data[0, 0, 0, :, 0, -1]

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(inputs['n_cumsteps'] / inputs['eval_freq'])
    count_z = int(inputs['n_trials'] )

    logtemp = np.zeros((envs, closs, count_x, count_z))

    for a in range(envs):
        for l in range(closs):
            for t in range(count_x):
                for n in range(inputs['n_trials']):
                    logtemp[a, l, t, n] = data[a, l, n, t, 0, 10]

    cols = ['C'+str(x) for x in range(closs)]

    env_name = [name[0:name.index("Bullet")] for name in env_name]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
    
    for sc in range(envs):
            for cl in range(closs):
            
                logtemp_cl = logtemp[sc, cl] 
                logtemp_cl = np.exp(logtemp_cl)

                logtemp_mean = np.mean(logtemp_cl, axis=1, keepdims=True)
                logtemp_max, logtemp_min = np.max(logtemp_cl, axis=1, keepdims=True), np.min(logtemp_cl, axis=1, keepdims=True)
                logtemp_mad = np.mean(np.abs(logtemp_cl - logtemp_mean), axis=1, keepdims=True)
                logtemp_mad_up = np.minimum(logtemp_max, logtemp_mean+logtemp_mad).reshape(-1)
                logtemp_mad_lo = np.maximum(logtemp_min, logtemp_mean-logtemp_mad).reshape(-1)
                logtemp_mean = logtemp_mean.reshape(-1)

                axs[sc].plot(x_steps, logtemp_mean, color=cols[cl], linewidth=1)
                axs[sc].fill_between(x_steps, logtemp_mad_lo, logtemp_mad_up, facecolor=cols[cl], alpha=0.1)
                axs[sc].grid(True, linewidth=0.2)
                axs[sc].set_title(env_name[sc])

    axs[0].set_ylabel('Entropy Temperature α')
    axs[0].set_xlabel('Steps (1e'+str(exp)+')')

    fig.subplots_adjust(bottom=0.3, hspace=0.2)
    fig.legend(critic_name, loc='lower center', ncol=closs, frameon=False, fontsize='medium')

    plt.savefig(filename_png, dpi=300, format='png')
0
def plot_inv(inputs: dict, reward: np.ndarray, lev: np.ndarray, stop: np.ndarray, reten: np.ndarray, loss: np.ndarray, 
             tail: np.ndarray, shadow: np.ndarray, cmax: np.ndarray, keqv: np.ndarray, filename_png: str, 
             T: int =1, V_0: float =1):
    """
    Plot summary of investors for a constant number of assets.

    Parameters:
        inputs: dictionary containing all execution details
        reward: 1 + time-average growth rate
        lev: leverages
        stop: stop-losses
        reten: retention ratios
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        filename_png (directory): save path of plot
        T: amount of compunding for reward
        V_0: intial value to compound
    """
    ninv = reward.shape[0]
    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    cols = ['C'+str(x) for x in range(ninv)]
    a_col = mpatches.Patch(color=cols[0], label='Inv A', alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label='Inv B', alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label='Inv C', alpha=0.8)

    reward = V_0 * reward**T

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8,10))

    for i in range(ninv):

        inv_x = reward[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean+x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean-x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        
        # x_mean = np.log10(x_mean)
        # x_med = np.log10(x_med)
        # x_mad_lo = np.log10(x_mad_lo)
        # x_mad_up = np.log10(x_mad_up)

        axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[0, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
        # axs[0, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
        axs[0, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
        # axs[0, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[0, 0].grid(True, linewidth=0.2)
        axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('Growth ' + r'$\bar{g}$' + ' (%)') 

    for i in range(ninv):

        inv_x = lev[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        print(x_med)
        print(x_mean)

        x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)

        axs[1, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[1, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
        # axs[1, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
        axs[1, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
        # axs[1, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[1, 0].grid(True, linewidth=0.2)
        axs[1, 0].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel('Leverage')

    for i in range(1, ninv):

        inv_x = stop[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100
        
        x_mean = x_mean.reshape(-1) * 100
        x_med = x_med.reshape(-1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
        x_d = x_d.reshape(-1) * 100
        x_u = x_u.reshape(-1) * 100

        axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[2, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
        # axs[2, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
        axs[2, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
        # axs[2, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[2, 0].grid(True, linewidth=0.2)
        axs[2, 0].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel('Stop-Loss ' + r'$\lambda$ ' + '(%)')

    for i in range(2, ninv):

        inv_x = reten[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100

        x_mean = x_mean.reshape(-1) * 100
        x_med = x_med.reshape(-1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
        x_d = x_d.reshape(-1) * 100
        x_u = x_u.reshape(-1) * 100

        axs[3, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[3, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
        # axs[3, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
        axs[3, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
        # axs[3, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[3, 0].grid(True, linewidth=0.2)

    axs[3, 0].set_ylabel('Retention ' + r'$\phi$ ' + '(%)')
    axs[3, 0].set_xlabel('Steps (1e'+str(exp)+')')

    for i in range(ninv):

        inv_x = loss[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        axs[0, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[0, 1].grid(True, linewidth=0.2)
        axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_ylabel('Critic')

    for i in range(ninv):

        inv_x = tail[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        axs[1, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[1, 1].grid(True, linewidth=0.2)
        axs[1, 1].xaxis.set_ticklabels([])

    axs[1, 1].set_ylabel('Critic Tail ' + r'$\alpha}$')

    for i in range(ninv):

        inv_x = shadow[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        axs[2, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

        inv_x = cmax[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=':')
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=':')
        # axs[2, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

    axs[2, 1].set_ylabel('Critic Shadow ' + r'$\mu_s$')

    for i in range(ninv):

        inv_x = keqv[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[3, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[3, 1].grid(True, linewidth=0.2)
        axs[3, 1].xaxis.set_ticklabels([])

    axs[3, 1].set_ylabel('Multiplier ' + r'$\kappa_{eqv}$')

    fig.subplots_adjust(bottom=0.1,  wspace=0.3, hspace=0.4)
    fig.legend(handles=[a_col, b_col, c_col], loc='lower center', ncol=3, frameon=False, fontsize='medium')

    plt.savefig(filename_png, dpi=300, format='png')
    
def plot_inv_all_n_perf(inputs: dict, reward_1: np.ndarray, lev_1: np.ndarray, stop_1: np.ndarray, reten_1: np.ndarray,
                        reward_2: np.ndarray, lev_2: np.ndarray, stop_2: np.ndarray, reten_2: np.ndarray,
                        reward_10: np.ndarray, lev_10: np.ndarray, stop_10: np.ndarray, reten_10: np.ndarray,
                        filename_png: str, T: int =1, V_0: float =1):
    """
    Plot summary of investor performance across three counts of assets N = 1, 2, 10.

    Parameters:
        inputs: dictionary containing all execution details
        reward_1: 1 + time-average growth rate for n=1 assets
        lev_1: leverages for n=1 assets
        stop_1: stop-losses for n=1 assets
        reten_1: retention ratios for n=1 assets
            ...
        filename_png (directory): save path of plot
        T: amount of compunding for reward
        V_0: intial value to compound
    """
    ninv = reward_1.shape[0]
    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    cols = ['C'+str(x) for x in range(ninv)]
    a_col = mpatches.Patch(color=cols[0], label='Inv A', alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label='Inv B', alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label='Inv C', alpha=0.8)

    reward_1, reward_2, reward_10 = V_0 * reward_1**T, V_0 * reward_2**T, V_0 * reward_10**T

    fig, axs = plt.subplots(nrows=4, ncols=ninv, figsize=(10,12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = reward_1[i]
            elif n == 1:
                inv_x = reward_2[i]
            else:
                inv_x = reward_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = (np.minimum(x_max, x_mean+x_mad).reshape(-1) - 1) * 100
            x_mad_lo = (np.maximum(x_min, x_mean-x_mad).reshape(-1) - 1) * 100

            x_mean = (x_mean.reshape(-1) - 1) * 100
            x_med = (x_med.reshape(-1) - 1) * 100

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = (x_d.reshape(-1) - 1) * 100
            x_u = (x_u.reshape(-1) - 1) * 100
            
            # x_mean = np.log10(x_mean)
            # x_med = np.log10(x_med)
            # x_mad_lo = np.log10(x_mad_lo)
            # x_mad_up = np.log10(x_mad_up)

            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[0, n].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[0, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[0, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('Growth ' + r'$\bar{g}$' + ' (%)') 

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = lev_1[i]
            elif n == 1:
                inv_x = lev_2[i]
            else:
                inv_x = lev_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)
            
            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[1, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[1, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel('Leverage')

    for n in range(3):
        for i in range(1, ninv, 1):

            if n == 0:
                inv_x = stop_1[i]
            elif n == 1:
                inv_x = stop_2[i]
            else:
                inv_x = stop_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[2, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[2, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel('Stop-Loss ' + r'$\lambda$ ' + '(%)')

    for n in range(3):
        for i in range(2, ninv, 1):

            if n == 0:
                inv_x = reten_1[i]
            elif n == 1:
                inv_x = reten_2[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = reten_10[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[3, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[3, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 0].set_ylabel('Retention ' + r'$\phi$ ' + '(%)')
    axs[3, 0].set_xlabel('Steps (1e'+str(exp)+')')

    axs[0, 0].text(0.35, 1.2, r'$N = 1$', size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.35, 1.2, r'$N = 2$', size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.35, 1.2, r'$N = 10$', size='large', transform=axs[0, 2].transAxes)

    fig.subplots_adjust(bottom=0.075,  wspace=0.25, hspace=0.3)
    fig.legend(handles=[a_col, b_col, c_col], loc='lower center', ncol=3, frameon=False, fontsize='medium')

    plt.savefig(filename_png, dpi=300, format='png')

def plot_inv_all_n_train(inputs: dict, loss_1: np.ndarray, tail_1: np.ndarray, shadow_1: np.ndarray, keqv1: np.ndarray,
                         loss_2: np.ndarray, tail_2: np.ndarray, shadow_2: np.ndarray, keqv2: np.ndarray,
                         loss_10: np.ndarray, tail_10: np.ndarray, shadow_10: np.ndarray, keqv10: np.ndarray,
                         filename_png: str):
    """
    Plot summary of investor training across three counts of assets N = 1, 2, 10.

    Parameters:
        inputs: dictionary containing all execution details
        loss_1: mean critic loss for n=1 assets
        tail_1: tail exponent for n=1 assets
        shadow_1: critic shadow loss for n=1 assets
        keqv_1: equivilance multiplier for n=1 assets
            ...
        filename_png (directory): save path of plot
        T: amount of compunding for reward
        V_0: intial value to compound
    """
    ninv = loss_1.shape[0]
    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    cols = ['C'+str(x) for x in range(ninv)]
    a_col = mpatches.Patch(color=cols[0], label='Inv A', alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label='Inv B', alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label='Inv C', alpha=0.8)

    fig, axs = plt.subplots(nrows=4, ncols=ninv, figsize=(10,12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = loss_1[i]
            elif n == 1:
                inv_x = loss_2[i]
            else:
                inv_x = loss_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[0, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            axs[0, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('Critic') 

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = tail_1[i]
            elif n == 1:
                inv_x = tail_2[i]
            else:
                inv_x = tail_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)
            
            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[1, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            axs[1, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel('Critic Tail ' + r'$\alpha}$')

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = shadow_1[i]
            elif n == 1:
                inv_x = shadow_2[i]
            else:
                inv_x = shadow_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[2, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            axs[2, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel('Critic Shadow ' + r'$\mu_s$')

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = keqv1[i]
            elif n == 1:
                inv_x = keqv2[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = keqv10[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            # axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[3, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[3, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 0].set_ylabel('Multiplier ' + r'$\kappa_{eqv}$')
    axs[3, 0].set_xlabel('Steps (1e'+str(exp)+')')

    axs[0, 0].text(0.35, 1.2, r'$N = 1$', size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.35, 1.2, r'$N = 2$', size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.35, 1.2, r'$N = 10$', size='large', transform=axs[0, 2].transAxes)

    fig.subplots_adjust(bottom=0.075,  wspace=0.25, hspace=0.3)
    fig.legend(handles=[a_col, b_col, c_col], loc='lower center', ncol=3, frameon=False, fontsize='medium')

    plt.savefig(filename_png, dpi=300, format='png')

def plot_safe_haven(inputs: dict, reward: np.ndarray, lev: np.ndarray, stop: np.ndarray, reten: np.ndarray, loss: np.ndarray, 
                    tail: np.ndarray, shadow: np.ndarray, cmax: np.ndarray, keqv: np.ndarray, lev_sh: np.ndarray,
                    filename_png: str, inv: str, T: int =1, V_0: float =1):
    """
    Plot summary of investors for safe haven.

    Parameters:
        inputs: dictionary containing all execution details
        reward: 1 + time-average growth rate
        lev: leverages
        stop: stop-losses
        reten: retention ratios
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        filename_png (directory): save path of plot
        inv: whether 'a', 'b' or 'c'
        T: amount of compunding for reward
        V_0: intial value to compound
    """
    ninv = reward.shape[0]
    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    if inv == 'a':
        inv_col = 'C0'
    elif inv == 'b':
        inv_col = 'C1'
    else:
        inv_col = 'C2'

    cols = [inv_col, 'C4']
    a_col = mpatches.Patch(color=cols[0], label='Uninsured', alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label='Insured', alpha=0.8)

    reward = V_0 * reward**T

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8,10))

    for i in range(ninv):

        inv_x = reward[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean+x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean-x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        
        # x_mean = np.log10(x_mean)
        # x_med = np.log10(x_med)
        # x_mad_lo = np.log10(x_mad_lo)
        # x_mad_up = np.log10(x_mad_up)

        axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[0, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
        # axs[0, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
        axs[0, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
        # axs[0, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[0, 0].grid(True, linewidth=0.2)
        axs[0, 0].xaxis.set_ticklabels([])

        print(x_med, x_mean)

    axs[0, 0].set_ylabel('Growth ' + r'$\bar{g}$' + ' (%)') 

    for i in range(ninv):

        inv_x = lev[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)
  
        axs[1, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[1, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
        # axs[1, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
        axs[1, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
        # axs[1, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[1, 0].grid(True, linewidth=0.2)

    axs[1, 0].set_ylabel('Leverage')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')

    if inv != 'a':
        for i in range(ninv):
            
                inv_x = stop[i]

                x_mean = np.mean(inv_x, axis=1, keepdims=True)
                x_med = np.median(inv_x, axis=1, keepdims=True)

                x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100

                x_mean = x_mean.reshape(-1) * 100
                x_med = x_med.reshape(-1) * 100

                x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
                x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
                x_d = x_d.reshape(-1) * 100
                x_u = x_u.reshape(-1) * 100

                axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
                axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
                # axs[2, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
                axs[2, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
                # axs[2, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
                axs[2, 0].grid(True, linewidth=0.2)

        axs[2, 0].set_ylabel('Stop-Loss ' + r'$\lambda$ ' + '(%)')

    if inv == 'c':
        for i in range(ninv):

                inv_x = reten[i]

                x_mean = np.mean(inv_x, axis=1, keepdims=True)
                x_med = np.median(inv_x, axis=1, keepdims=True)

                x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100

                x_mean = x_mean.reshape(-1) * 100
                x_med = x_med.reshape(-1) * 100

                x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
                x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
                x_d = x_d.reshape(-1) * 100
                x_u = x_u.reshape(-1) * 100

                axs[3, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
                axs[3, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
                # axs[3, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
                axs[3, 0].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
                # axs[3, i].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
                axs[3, 0].grid(True, linewidth=0.2)

        axs[3, 0].set_ylabel('Retention ' + r'$\phi$ ' + '(%)')

    for i in range(ninv):

        inv_x = loss[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        axs[0, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[0, 1].grid(True, linewidth=0.2)
        axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_ylabel('Critic')

    for i in range(ninv):

        inv_x = tail[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        axs[1, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[1, 1].grid(True, linewidth=0.2)
        axs[1, 1].xaxis.set_ticklabels([])

    axs[1, 1].set_ylabel('Critic Tail ' + r'$\alpha}$')

    for i in range(ninv):

        inv_x = shadow[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        axs[2, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

        inv_x = cmax[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=':')
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=':')
        # axs[2, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

    axs[2, 1].set_ylabel('Critic Shadow ' + r'$\mu_s$')

    for i in range(ninv):

        inv_x = keqv[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.median(inv_x, axis=1, keepdims=True)

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
        
        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
        # axs[3, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[3, 1].grid(True, linewidth=0.2)
        axs[3, 1].xaxis.set_ticklabels([])

    axs[3, 1].set_ylabel('Multiplier ' + r'$\kappa_{eqv}$')

    if inv == 'a':
        axs[2, 0].set_axis_off()
        axs[3, 0].set_axis_off()
    elif inv == 'b':
        axs[3, 0].set_axis_off()
        axs[1, 0].xaxis.set_ticklabels([])
    else:
        axs[1, 0].xaxis.set_ticklabels([])
        axs[2, 0].xaxis.set_ticklabels([])


    fig.subplots_adjust(bottom=0.1,  wspace=0.3, hspace=0.4)
    fig.legend(handles=[a_col, b_col], loc='lower center', ncol=3, frameon=False, fontsize='medium')

    plt.savefig(filename_png, dpi=300, format='png')

def plot_inv_sh_perf(inputs: dict, reward_a: np.ndarray, lev_a: np.ndarray, stop_a: np.ndarray, reten_a: np.ndarray, levsh_a: np.ndarray,
                     reward_b: np.ndarray, lev_b: np.ndarray, stop_b: np.ndarray, reten_b: np.ndarray, levsh_b: np.ndarray,
                     reward_c: np.ndarray, lev_c: np.ndarray, stop_c: np.ndarray, reten_c: np.ndarray, levsh_c: np.ndarray,
                     filename_png: str, T: int =1, V_0: float =1):
    """
    Plot summary of investor performance across three counts of assets N = 1, 2, 10.

    Parameters:
        inputs: dictionary containing all execution details
        reward_a: 1 + time-average growth rate for invA with and without safe haven 
        lev_a: leverages for invA with and without safe haven
        stop_a: stop-losses for invA with and without safe haven
        reten_a: retention ratios for invA with and without safe haven
        levsh_a: safe haven leverage for invA with and without safe haven
            ...
        filename_png (directory): save path of plot
        T: amount of compunding for reward
        V_0: intial value to compound
    """
    ninv = reward_a.shape[0]

    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    cols = ['C0', 'C4']
    a_col = mpatches.Patch(color=cols[0], label='Uninsured', alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label='Insured', alpha=0.8)
    c_col = mpatches.Patch(color='C3', label='Safe Haven', alpha=0.8)

    reward_a, reward_b, reward_c = V_0 * reward_a**T, V_0 * reward_b**T, V_0 * reward_c**T

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10,12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = reward_a[i]
            elif n == 1:
                inv_x = reward_b[i]
            else:
                inv_x = reward_c[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = (np.minimum(x_max, x_mean+x_mad).reshape(-1) - 1) * 100
            x_mad_lo = (np.maximum(x_min, x_mean-x_mad).reshape(-1) - 1) * 100

            x_mean = (x_mean.reshape(-1) - 1) * 100
            x_med = (x_med.reshape(-1) - 1) * 100

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = (x_d.reshape(-1) - 1) * 100
            x_u = (x_u.reshape(-1) - 1) * 100
            
            # x_mean = np.log10(x_mean)
            # x_med = np.log10(x_med)
            # x_mad_lo = np.log10(x_mad_lo)
            # x_mad_up = np.log10(x_mad_up)
            # print(x_mean)
            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[0, n].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=':')
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[0, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[0, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('Growth ' + r'$\bar{g}$' + ' (%)') 

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = lev_a[i]
            elif n == 1:
                inv_x = lev_b[i]
            else:
                inv_x = lev_c[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)
            
            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[1, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[1, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

            if n == 0:
                inv_x = levsh_a[i]
            elif n == 1:
                inv_x = levsh_b[i]
            else:
                inv_x = levsh_c[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)
            
            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color='C3', linewidth=1)
            axs[1, n].plot(x_steps, x_med, color='C3', linewidth=1, linestyle='--')
            # axs[1, n].plot(x_steps, x_u, color='C3', linewidth=1, linestyle=':')
            axs[1, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor='C3', edgecolor='C3', linewidth=2, linestyle='--')
            # axs[1, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel('Leverage')

    for n in range(3):
        for i in range(0, ninv, 1):

            if n == 0:
                inv_x = stop_a[i]
            elif n == 1:
                inv_x = stop_b[i]
            else:
                inv_x = stop_c[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[2, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[2, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 1].set_ylabel('Stop-Loss ' + r'$\lambda$ ' + '(%)')

    for n in range(3):
        for i in range(0, 2, 1):

            if n == 0:
                inv_x = reten_a[i]
            elif n == 1:
                inv_x = reten_b[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = reten_c[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            axs[3, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[3, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 2].set_ylabel('Retention ' + r'$\phi$ ' + '(%)')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')


    axs[2, 0].set_axis_off()
    axs[3, 0].set_axis_off()
    axs[3, 1].set_axis_off()

    # axs[2, 0].xaxis.set_ticklabels([])
    # axs[3, 0].xaxis.set_ticklabels([])
    # axs[3, 1].xaxis.set_ticklabels([])

    axs[0, 0].text(0.375, 1.2, 'Inv A', size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.375, 1.2, 'Inv B', size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.375, 1.2, 'Inv C', size='large', transform=axs[0, 2].transAxes)

    fig.subplots_adjust(bottom=0.075,  wspace=0.25, hspace=0.3)
    fig.legend(handles=[a_col, b_col, c_col], loc='lower center', ncol=3, frameon=False, fontsize='medium')

    plt.savefig(filename_png, dpi=300, format='png')

def plot_inv_sh_train(inputs: dict, loss_1: np.ndarray, tail_1: np.ndarray, shadow_1: np.ndarray, keqv1: np.ndarray,
                      loss_2: np.ndarray, tail_2: np.ndarray, shadow_2: np.ndarray, keqv2: np.ndarray,
                      loss_10: np.ndarray, tail_10: np.ndarray, shadow_10: np.ndarray, keqv10: np.ndarray,
                      filename_png: str):
    """
    Plot summary of investor training across three counts of assets N = 1, 2, 10.

    Parameters:
        inputs: dictionary containing all execution details
        loss_1: mean critic loss for n=1 assets
        tail_1: tail exponent for n=1 assets
        shadow_1: critic shadow loss for n=1 assets
        keqv_1: equivilance multiplier for n=1 assets
            ...
        filename_png (directory): save path of plot
        T: amount of compunding for reward
        V_0: intial value to compound
    """
    ninv = loss_1.shape[0]
    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    cols = ['C0', 'C4']
    a_col = mpatches.Patch(color=cols[0], label='Uninsured', alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label='Insured', alpha=0.8)

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10,12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = loss_1[i]
            elif n == 1:
                inv_x = loss_2[i]
            else:
                inv_x = loss_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[0, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            axs[0, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('Critic') 

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = tail_1[i]
            elif n == 1:
                inv_x = tail_2[i]
            else:
                inv_x = tail_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)
            
            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[1, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            axs[1, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel('Critic Tail ' + r'$\alpha}$')

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = shadow_1[i]
            elif n == 1:
                inv_x = shadow_2[i]
            else:
                inv_x = shadow_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[2, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            axs[2, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel('Critic Shadow ' + r'$\mu_s$')

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = keqv1[i]
            elif n == 1:
                inv_x = keqv2[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = keqv10[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.median(inv_x, axis=1, keepdims=True)

            x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(inv_x, 25, axis=1, interpolation='lower', keepdims=True)
            x_u = np.percentile(inv_x, 75, axis=1, interpolation='lower', keepdims=True)
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            # axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle='--')
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=':')
            # axs[3, n].fill_between(x_steps, x_d, x_u, alpha=0.1, facecolor=cols[i], edgecolor=cols[i], linewidth=2, linestyle='--')
            # axs[3, n].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 0].set_ylabel('Multiplier ' + r'$\kappa_{eqv}$')
    axs[3, 0].set_xlabel('Steps (1e'+str(exp)+')')

    axs[0, 0].text(0.375, 1.2, 'Inv A', size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.375, 1.2, 'Inv B', size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.375, 1.2, 'Inv C', size='large', transform=axs[0, 2].transAxes)
    fig.subplots_adjust(bottom=0.075,  wspace=0.25, hspace=0.3)
    fig.legend(handles=[a_col, b_col], loc='lower center', ncol=3, frameon=False, fontsize='medium')

    plt.savefig(filename_png, dpi=300, format='png')