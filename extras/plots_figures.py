#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  plot_figures.py
python version:         3.9
torch verison:          1.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Plotting of all final summary figures for all reinforcement learning experiments. 
"""

import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm
import torch as T
from typing import List

import extras.critic_loss as closs
import extras.utils as utils

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

    mse = closs.mse(a, b, 0)
    mse2 = closs.mse(a, b, 2)
    mse4 = closs.mse(a, b, 4)
    huber = closs.huber(a, b)
    mae = closs.mae(a, b)
    hsc = closs.hypersurface(a, b)
    cauchy = closs.cauchy(a, b, 1)

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
    plt.title('Loss', size='large')
    plt.legend(l, loc='lower right', ncol=1, frameon=False, fontsize='medium', markerscale=6)
    plt.tight_layout()

    plt.savefig(filename_png, dpi=200, format='png')

def plot_gbm_max_lev(filename_png: str):
    """
    Plot of GBM maximum leverage across down sigma moves.

    Parameters:
        filename_png (directory): save path of plot
    """
    # S&P500 parameters
    mu = 0.0516042410820218
    sigma = 0.190381677824107
    v0 = 2
    vmin = 1

    sig = np.linspace(1, 10, 1000)
    prob = norm.pdf(sig, 0, 1)
    prob = np.log10(prob)

    rho = np.linspace(1, 10, 1000)

    l_eff = 2 / (sigma * (2 * rho + sigma) - 2 * mu) * np.log(v0 / vmin)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='lev')
    ax2 = fig.add_subplot(1,1,1, label='prob', frame_on=False)

    ax1.plot(rho, l_eff, color='C0')
    ax1.set_xlabel(r'$\rho_d$')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Maximum Leverage', color='C0')
    ax1.yaxis.set_label_position('left')
    ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, linewidth=0.5)

    ax2.plot(sig, prob, color='C3')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Probability (log10)', color='C3')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C3')
    
    plt.savefig(filename_png, dpi=200, format='png')

def plot_add(inputs: dict, env_names: List[str], legend: List[str], multi: bool, 
             reward: np.ndarray, loss: np.ndarray, scale: np.ndarray, kernel: np.ndarray, 
             tail: np.ndarray, shadow: np.ndarray, keqv: np.ndarray, filename_png: str):
    """
    Plots additve environments figures for loss functions and multi-step returns.

    Parameters:
        inputs: dictionary containing all execution details
        env_names: list of environment names
        legend: list of labeling across trials
        multi: True or False as to whther plotting multi-stewp returns
        rewards: rewards across trials
        loss: critic losses across trials
        scale: Cauchy scales across trials
        kernel: CIM kernel sizes across trials
        tail: critic tail exponents across trials
        shadow: critic shahow means across trials
        keqv: multiplier for equvilance between shadow and empirical means across trials
        filename_png: path for file saving
    """
    n_env, n_algo, n_data = reward.shape[0], reward.shape[1], reward.shape[2]

    if multi == True:
        legend = ['m = '+str(legend[x]) for x in range(n_data)]

    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    cols = ['C'+str(x) for x in range(n_data)]

    patches = [mpatches.Patch(color=cols[x], label=legend[x], alpha=0.8)
               for x in range(n_data)]

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = reward[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max, x_min = np.max(var_x, axis=1, keepdims=True), np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
                
                x_mean = x_mean.reshape(-1)

                axs[a, e].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[a, e].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('SAC Score')
    axs[1, 0].set_ylabel('TD3 Score')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.325, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.325, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.325, 1.1, env_names[2], size='large', transform=axs[0, 2].transAxes)
    axs[0, 3].text(0.325, 1.1, env_names[3], size='large', transform=axs[0, 3].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_score.png', dpi=200, format='png')

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = loss[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max, x_min = np.max(var_x, axis=1, keepdims=True), np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
                
                x_mean = x_mean.reshape(-1)
       
                x_mean, x_mad_up, x_mad_lo = np.log10(x_mean), np.log10(x_mad_up), np.log10(x_mad_lo)

                axs[a, e].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[a, e].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('SAC Critic (log10)')
    axs[1, 0].set_ylabel('TD3 Critic (log10)')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.325, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.325, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.325, 1.1, env_names[2], size='large', transform=axs[0, 2].transAxes)
    axs[0, 3].text(0.325, 1.1, env_names[3], size='large', transform=axs[0, 3].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_loss.png', dpi=200, format='png')

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = scale[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max, x_min = np.max(var_x, axis=1, keepdims=True), np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
                
                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = np.log10(x_mean), np.log10(x_mad_up), np.log10(x_mad_lo)

                axs[a, e].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[a, e].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('SAC Cauchy Scale ' + r'$\gamma$' + ' (log10)')
    axs[1, 0].set_ylabel('TD3 Cauchy Scale ' + r'$\gamma$' + ' (log10)')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.325, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.325, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.325, 1.1, env_names[2], size='large', transform=axs[0, 2].transAxes)
    axs[0, 3].text(0.325, 1.1, env_names[3], size='large', transform=axs[0, 3].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_scale.png', dpi=200, format='png')

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = kernel[e, a, d]

                # use np.nan conversion to ignore divergences
                x_mean = np.nanmean(var_x, axis=1, keepdims=True)

                x_max, x_min = np.nanmax(var_x, axis=1, keepdims=True), np.nanmin(var_x, axis=1, keepdims=True)
                x_mad = np.nanmean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = np.log10(x_mean), np.log10(x_mad_up), np.log10(x_mad_lo)  

                axs[a, e].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[a, e].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('SAC CIM Kernel ' + r'$\sigma$' + ' (log10)')
    axs[1, 0].set_ylabel('TD3 CIM Kernel ' + r'$\sigma$' + ' (log10)')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.325, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.325, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.325, 1.1, env_names[2], size='large', transform=axs[0, 2].transAxes)
    axs[0, 3].text(0.325, 1.1, env_names[3], size='large', transform=axs[0, 3].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_kernel.png', dpi=200, format='png')

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = tail[e, a, d]
                
                # use np.nan conversion to ignore divergences
                x_mean = np.nanmean(var_x, axis=1, keepdims=True)

                x_max, x_min = np.nanmax(var_x, axis=1, keepdims=True), np.nanmin(var_x, axis=1, keepdims=True)
                x_mad = np.nanmean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
                
                x_mean = x_mean.reshape(-1)

                axs[a, e].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[a, e].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('SAC Critic Tail ' + r'$\alpha$')
    axs[1, 0].set_ylabel('TD3 Critic Tail ' + r'$\alpha$')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.325, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.325, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.325, 1.1, env_names[2], size='large', transform=axs[0, 2].transAxes)
    axs[0, 3].text(0.325, 1.1, env_names[3], size='large', transform=axs[0, 3].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_tail.png', dpi=200, format='png')

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = shadow[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max, x_min = np.max(var_x, axis=1, keepdims=True), np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
                
                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = np.log10(x_mean), np.log10(x_mad_up), np.log10(x_mad_lo)

                axs[a, e].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[a, e].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('SAC Critic Shadow ' + r'$\mu_s$' + ' (log10)')
    axs[1, 0].set_ylabel('TD3 Critic Shadow ' + r'$\mu_s$' + ' (log10)')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.325, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.325, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.325, 1.1, env_names[2], size='large', transform=axs[0, 2].transAxes)
    axs[0, 3].text(0.325, 1.1, env_names[3], size='large', transform=axs[0, 3].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_shadow.png', dpi=200, format='png')

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = keqv[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max, x_min = np.max(var_x, axis=1, keepdims=True), np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
                
                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = np.log10(x_mean), np.log10(x_mad_up), np.log10(x_mad_lo)

                axs[a, e].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[a, e].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel('SAC Multiplier ' + r'$\kappa_{eqv}$' + ' (log10)')
    axs[1, 0].set_ylabel('TD3 Multiplier ' + r'$\kappa_{eqv}$' + ' (log10)')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.325, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.325, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.325, 1.1, env_names[2], size='large', transform=axs[0, 2].transAxes)
    axs[0, 3].text(0.325, 1.1, env_names[3], size='large', transform=axs[0, 3].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_keqv.png', dpi=200, format='png')

def plot_add_temp(inputs: dict, env_names: List[str], legend: List[str], multi: bool, 
                  logtemp: np.ndarray, filename_png: str):
    """
    Plots additve environments SAC entropy temperature for loss functions and multi-step returns.

    Parameters:
        inputs: dictionary containing all execution details
        env_names: list of environment names
        legend: list of labeling across trials
        multi: True or False as to whther plotting multi-stewp returns
        logtemp: log SAC entopy temperature across trials
        filename_png: path for file saving
    """
    n_env, n_algo, n_data = logtemp.shape[0], logtemp.shape[1], logtemp.shape[2]

    if multi == True:
        legend = ['m = '+str(legend[x]) for x in range(n_data)]

    cum_steps_log = np.array([x for x in range(int(inputs['eval_freq']), int(inputs['n_cumsteps']) + 
                                               int(inputs['eval_freq']), int(inputs['eval_freq']))])

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    cols = ['C'+str(x) for x in range(n_data)]

    patches = [mpatches.Patch(color=cols[x], label=legend[x], alpha=0.8)
               for x in range(n_data)]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

    for e in range(n_env):
        for d in range(n_data):

            var_x = logtemp[e, 0, d]

            var_x = np.exp(var_x)

            x_mean = np.mean(var_x, axis=1, keepdims=True)

            x_max, x_min = np.max(var_x, axis=1, keepdims=True), np.min(var_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)
            
            x_mean = x_mean.reshape(-1)

            x_mean, x_mad_up, x_mad_lo = np.log10(x_mean), np.log10(x_mad_up), np.log10(x_mad_lo)

            if e == 0:
                axs[0, 0].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[0, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[0, 0].grid(True, linewidth=0.2)
                axs[0, 0].xaxis.set_ticklabels([])
            elif e == 1:
                axs[0, 1].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[0, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[0, 1].grid(True, linewidth=0.2)
                axs[0, 1].xaxis.set_ticklabels([])
            elif e == 2:
                axs[1, 0].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[1, 0].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[1, 0].grid(True, linewidth=0.2)
            else:
                axs[1, 1].plot(x_steps, x_mean, color=cols[d], linewidth=0.5)
                axs[1, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[d], alpha=0.15)
                axs[1, 1].grid(True, linewidth=0.2)
                axs[1, 1].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel('Entropy Temperature (log10)')
    axs[1, 0].set_xlabel('Steps (1e'+str(exp)+')')
    
    axs[0, 0].text(0.35, 1.1, env_names[0], size='large', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.35, 1.1, env_names[1], size='large', transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.35, 1.1, env_names[2], size='large', transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.35, 1.1, env_names[3], size='large', transform=axs[1, 1].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.15)

    if multi == False:
        fig.legend(handles=patches, loc='lower center',ncol=int(n_data/2 + 1), frameon=False, fontsize='large')
    else:
        fig.legend(handles=patches, loc='lower center',ncol=n_data, frameon=False, fontsize='large')

    plt.savefig(filename_png+'_temp.png', dpi=200, format='png')

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

    axs[1, 1].set_ylabel('Critic Tail ' + r'$\alpha$')

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

    plt.savefig(filename_png, dpi=200, format='png')
    
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

    plt.savefig(filename_png, dpi=200, format='png')

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

    axs[1, 0].set_ylabel('Critic Tail ' + r'$\alpha$')

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

    plt.savefig(filename_png, dpi=200, format='png')

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

    axs[1, 1].set_ylabel('Critic Tail ' + r'$\alpha$')

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

    plt.savefig(filename_png, dpi=200, format='png')

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

    plt.savefig(filename_png, dpi=200, format='png')

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

    axs[1, 0].set_ylabel('Critic Tail ' + r'$\alpha$')

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

    plt.savefig(filename_png, dpi=200, format='png')