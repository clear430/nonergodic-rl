import matplotlib.pyplot as plt
import numpy as np
import torch as T
from typing import List

def plot_inv1(inv1_data: np.ndarray, inv1_data_T: np.ndarray, filename_png: str):
    """
    Investor 1 grid of plots.

    Parameters:
        inv1_data: array of investor 1 data across all fixed leverages
        inv1_data_T: array of investor 1 valutations at maturity 
        filename_png (directory): save path of plot
    """
    nlevs = inv1_data.shape[0]
    investors = inv1_data_T.shape[1]
    levs = (inv1_data[:, 9, 0] * 100).tolist()
    levs = [str(int(levs[l]))+'%' for l in range(nlevs)]
    x_steps = np.arange(0, inv1_data.shape[2])
    cols = ['C'+str(x) for x in range(nlevs)]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    
    for lev in range(nlevs):
            
            mean_adj_v = inv1_data[lev, 2].reshape(-1)
            mean_adj_v = np.log10(mean_adj_v)
            axs[0, 0].plot(x_steps, mean_adj_v, color=cols[lev], linewidth=1)
            axs[0, 0].grid(True, linewidth=0.2)
    
    axs[0, 0].set_ylabel('Adjusted Mean (log10)', size='small')
    axs[0, 0].text(0.5, -0.25, "(a)", size='small', transform=axs[0, 0].transAxes)
    axs[0, 0].set_xlabel('Steps', size='small')

    vals = [inv1_data_T[:, lev] for lev in range(investors)]
    vals = np.log10(vals)

    axs[0, 1].boxplot(vals, levs, labels = [(x + 1) * 10 for x in range(nlevs)])
    axs[0, 1].grid(True, linewidth=0.2)
    axs[0, 1].set_xlabel('Leverage (%)', size='small')
    axs[0, 1].text(0.5, -0.25, "(b)", size='small', transform=axs[0, 1].transAxes)
    
    for lev in range(nlevs):
            
            mean_adj_v = inv1_data[lev, 2].reshape(-1)
            top_adj_v = inv1_data[lev, 1].reshape(-1)

            mean_adj_v, top_adj_v = np.log10(mean_adj_v), np.log10(top_adj_v)
            axs[1, 0].plot(top_adj_v, mean_adj_v, color=cols[lev], linewidth=0.2)
            axs[1, 0].grid(True, linewidth=0.2)
    
    axs[1, 0].set_ylabel('Adjusted Mean (log10)', size='small')
    axs[1, 0].text(0.5, -0.25, "(c)", size='small', transform=axs[1, 0].transAxes)
    axs[1, 0].set_xlabel('Top Mean (log10)', size='small')

    nor_mean, top_mean, adj_mean = inv1_data[:, 0, -1], inv1_data[:, 1, -1], inv1_data[:, 2, -1]
    nor_mad, top_mad, adj_mad = inv1_data[:, 3, -1], inv1_data[:, 4, -1], inv1_data[:, 5, -1]
    nor_std, top_std, adj_std = inv1_data[:, 6, -1], inv1_data[:, 7, -1], inv1_data[:, 8, -1]

    max = 1e30
    min = 1e-39

    nor_mad_up, top_mad_up, adj_mad_up = np.minimum(max, nor_mean+nor_mad), np.minimum(max, top_mean+top_mad), np.minimum(max, adj_mean+adj_mad)
    nor_mad_lo, top_mad_lo, adj_mad_lo = np.maximum(min, nor_mean-nor_mad), np.maximum(min, top_mean-top_mad), np.maximum(min, adj_mean-adj_mad)

    nor_std_up, top_std_up,  adj_std_up = np.minimum(max, nor_mean+nor_std), np.minimum(max, top_mean+top_std), np.minimum(max, adj_mean+adj_std)

    nor_mean, top_mean, adj_mean = np.log10(nor_mean), np.log10(top_mean), np.log10(adj_mean)
    nor_mad_up, top_mad_up, adj_mad_up =  np.log10(nor_mad_up), np.log10(top_mad_up), np.log10(adj_mad_up)
    nor_mad_lo, top_mad_lo, adj_mad_lo =  np.log10(nor_mad_lo), np.log10(top_mad_lo), np.log10(adj_mad_lo)
    nor_std_up, top_std_up, adj_std_up =  np.log10(nor_std_up), np.log10(top_std_up), np.log10(adj_std_up)

    x = [(x + 1) * 10 for x in range(nlevs)]

    # axs[1, 1].plot(x, top_mean, color='r', linestyle='--', linewidth=0.25)
    axs[1, 1].fill_between(x, top_mean, top_mad_up, facecolor='r', alpha=0.75)
    axs[1, 1].fill_between(x, top_mean, top_std_up, facecolor='r', alpha=0.25)

    # axs[1, 1].plot(x, nor_mean, color='g', linestyle='--', linewidth=0.25)
    axs[1, 1].fill_between(x, nor_mean, nor_mad_up, facecolor='g', alpha=0.75)
    axs[1, 1].fill_between(x, nor_mean, nor_std_up, facecolor='g', alpha=0.25)

    # axs[1, 1].plot(x, adj_mean, color='b', linestyle='--', linewidth=1)
    axs[1, 1].fill_between(x, adj_mean, adj_mad_up, facecolor='b', alpha=0.75)
    # axs[1, 1].fill_between(x, adj_mean, adj_mad_lo, facecolor='b', alpha=0.75)
    axs[1, 1].fill_between(x, adj_mean, adj_std_up, facecolor='b', alpha=0.25)
    
    axs[1, 1].grid(True, linewidth=0.2)
    axs[1, 1].set_xlabel('Leverage (%)', size='small')
    axs[1, 1].text(0.5, -0.25, "(d)", size='small', transform=axs[1, 1].transAxes)

    fig.subplots_adjust(hspace=0.3)
    fig.legend(levs, loc='upper center', ncol=5, frameon=False, fontsize='medium', title='Leverage', title_fontsize='medium')

    axs[0, 0].tick_params(axis='both', which='major', labelsize='small')
    axs[0, 1].tick_params(axis='both', which='major', labelsize='small')
    axs[1, 0].tick_params(axis='both', which='major', labelsize='small')
    axs[1, 1].tick_params(axis='both', which='major', labelsize='small')

    plt.savefig(filename_png, dpi=1000, format='png')
    
def plot_inv2(inv2_data: np.ndarray, filename_png: str):
    """
    Investor 2 mean values and mean leverages.

    Parameters:
        inv2_data: array of investor 2 data for a single stop-loss
        filename_png (directory): save path of plot
    """
    x_steps = np.arange(0, inv2_data.shape[3])
    cols = ['C'+str(x) for x in range(3)]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    
    max_x = 30

    for sub in range(3):
            
            val = inv2_data[0, 0, sub, :max_x].reshape(-1)
            val = np.log10(val)

            axs[0].plot(x_steps[:max_x], val, color=cols[sub], linewidth=1)
            axs[0].grid(True, linewidth=0.2)
    
    axs[0].set_ylabel('Mean (log10)', size='small')
    axs[0].text(0.5, -0.40, "(a)", size='small', transform=axs[0].transAxes)
    axs[0].set_xlabel('Steps', size='small')

    for sub in range(3):
            
            lev = inv2_data[0, 0, 9 + sub, :max_x].reshape(-1)

            axs[1].plot(x_steps[:max_x], lev, color=cols[sub], linewidth=1)
            axs[1].grid(True, linewidth=0.2)
    
    axs[1].set_ylabel('Leverage', size='small')
    axs[1].text(0.5, -0.40, "(b)", size='small', transform=axs[1].transAxes)


    fig.subplots_adjust(bottom=0.3)
    fig.legend(['Complete Sample', 'Top Sample', 'Adjusted Sample'], loc='upper center', ncol=3, frameon=False, fontsize='medium')

    axs[0].tick_params(axis='both', which='major', labelsize='small')
    axs[1].tick_params(axis='both', which='major', labelsize='small')

    plt.savefig(filename_png, dpi=1000, format='png')

def plot_inv3(inv3_data: np.ndarray, filename_png: str):
    """
    Investor 3 density plot.

    Parameters:
        inv3_data: array of investor 3 data across all stop-losses and retention ratios
        filename_png (directory): save path of plot
    """
    roll = inv3_data[:, 0, 19, 0]
    stop = inv3_data[0, :, 18, 0]

    r_len = roll.shape[0]
    s_len = stop.shape[0]
    adj_mean = np.zeros((r_len, s_len))

    for r in range(r_len):
        for s in range(s_len):
            adj_mean[r, s] = inv3_data[r, s, 2, -1] 

    adj_mean = np.log10(adj_mean)
    
    plt.pcolormesh(stop*100, roll*100, adj_mean, shading='gouraud', vmin=adj_mean.min(), vmax=adj_mean.max())
    plt.colorbar()

    plt.tick_params(axis='both', which='major', labelsize='small')
    plt.ylabel('Retention Ratio Φ (%)', size='small')
    plt.xlabel('Stop-Loss λ (%)', size='small')
    plt.title('Adjusted Mean (log10)', size='medium')

    plt.savefig(filename_png, dpi=1000, format='png')