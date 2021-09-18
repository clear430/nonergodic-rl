import sys
sys.path.append("./")

import extras.utils as utils
import matplotlib.pyplot as plt
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

def plot_critic_2d(input_dict: dict, data: np.ndarray, algo_name: List[str], critic_name: List[str], filename_png: str):
    """
    2D plot of Mean, MAD, and STD of scores and twin critic loss during evaluation episodes for all trials in environment.
    
    Parameters:
        input_dict: dictionary containing all execution details
        eval_log: log of episode data for all trials
        filename_png (directory): save path of plot
    """
    algos = data.shape[0]
    closs = data.shape[1]
    cum_steps_log = data[0, 0, 0, :, 0, -1]

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * int(input_dict['n_eval']))

    scores = np.zeros((algos, closs, count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((algos, closs, count_x, count_y)) 

    for a in range(algos):
        for l in range(closs):
            for t in range(count_x):
                for n in range(input_dict['n_trials']):
                    for s in range(int(input_dict['n_eval'])):
                        scores[a, l, t, s + n * int(input_dict['n_eval'])] = data[a, l, n, t, s, 1]
                        loss[a, l, t, s + n * int(input_dict['n_eval'])] = np.mean(data[a, l, n, t, s, 3:5])

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    cols = ['C'+str(x) for x in range(closs)]
    name = input_dict['env_id']
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

    plt.savefig(filename_png, dpi=400, format='png')

def plot_critic_loss(input_dict: dict, data: np.ndarray, algo_name: List[str], critic_name: List[str], filename_png: str):
    """
    Plot of score, critic losses Cauchy scale, and CIM kernel for two algorithms and all critic loss functions 
    for a single environments.

    Parameters:
        input_dict: dictionary containing all execution details
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

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * int(input_dict['n_eval']))
    count_z = int(input_dict['n_trials'] )

    scores = np.zeros((algos, closs, count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((algos, closs, count_x, count_z * 2))
    scale = np.zeros((algos, closs, count_x, count_z * 2)) 
    kernel = np.zeros((algos, closs, count_x, count_z * 2)) 

    for a in range(algos):
        for l in range(closs):
            for t in range(count_x):
                for n in range(input_dict['n_trials']):

                    loss[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 3:5]
                    scale[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 11:13]
                    kernel[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 13:15]

                    for s in range(int(input_dict['n_eval'])):
                        scores[a, l, t, s + n * int(input_dict['n_eval'])] = data[a, l, n, t, s, 1]

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    cols = ['C'+str(x) for x in range(closs)]
    name = input_dict['env_id']
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

    plt.savefig(filename_png, dpi=400, format='png')

def plot_critic_shadow(input_dict: dict, data: np.ndarray, algo_name: List[str], critic_name: List[str], filename_png: str):
    """
    Plot of shadow losses and tail index for two algorithms and all critic loss functions for a single environments.

    Parameters:
        input_dict: dictionary containing all execution details
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

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * int(input_dict['n_eval']))
    count_z = int(input_dict['n_trials'] )

    scores = np.zeros((algos, closs, count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((algos, closs, count_x, count_z * 2))
    shadow = np.zeros((algos, closs, count_x, count_z * 2)) 
    alpha = np.zeros((algos, closs, count_x, count_z * 2)) 

    for a in range(algos):
        for l in range(closs):
            for t in range(count_x):
                for n in range(input_dict['n_trials']):

                    # loss[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 3:5]
                    shadow[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 5:7]
                    alpha[a, l, t, (n * 2):(n * 2) + 2 ] = data[a, l, n, t, 0, 7:9]

                    # for s in range(int(input_dict['n_eval'])):
                    #     scores[a, l, t, s + n * int(input_dict['n_eval'])] = data[a, l, n, t, s, 1]

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    cols = ['C'+str(x) for x in range(closs)]
    name = input_dict['env_id']
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

    plt.savefig(filename_png, dpi=400, format='png')

def plot_temp(input_dict: dict, data: np.ndarray, env_name: List[str], critic_name: List[str], filename_png: str):
    """
    Plot SAC sutomatically tuned temperature hyperparameters for only two environments and all critic loss functions.

    Parameters:
        input_dict: dictionary containing all execution details
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

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_z = int(input_dict['n_trials'] )

    logtemp = np.zeros((envs, closs, count_x, count_z))

    for a in range(envs):
        for l in range(closs):
            for t in range(count_x):
                for n in range(input_dict['n_trials']):
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

    plt.savefig(filename_png, dpi=400, format='png')