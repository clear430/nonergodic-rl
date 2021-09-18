import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
    levs = (inv1_data[:, -1, 0] * 100).tolist()
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

    plt.savefig(filename_png, dpi=400, format='png')
    
def plot_inv2(inv2_data: np.ndarray, filename_png: str):
    """
    Investor 2 mean values and mean leverages.

    Parameters:
        inv2_data: array of investor 2 data for a single stop-loss
        filename_png (directory): save path of plot
    """
    x_steps = np.arange(0, inv2_data.shape[3])
    cols = ['C'+str(x) for x in range(3)]

    avg_col = mpatches.Patch(color=cols[0], label='Complete Sample', alpha=0.8)
    top_col = mpatches.Patch(color=cols[1], label='Top Sample', alpha=0.8)
    adj_col = mpatches.Patch(color=cols[2], label='Adjusted Sample', alpha=0.8)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    
    max_x = 30         # maximum number of time steps to be plotted
    value_0 = 1e2      # intial portfolio value of each investor

    for sub in range(3):
            val = inv2_data[0, 0, sub, :max_x].reshape(-1)
            mad = inv2_data[0, 0, 3 + sub, :max_x].reshape(-1)
            mad_up = (val + mad).reshape(-1)
            mad_lo = np.maximum(inv2_data[0, 0, -2, 0] * value_0, val - mad).reshape(-1)
            med_val = inv2_data[0, 0, 9 + sub, :max_x].reshape(-1)

            val, mad_up, mad_lo, med_val = np.log10(val), np.log10(mad_up), np.log10(mad_lo), np.log10(med_val)

            axs[0].plot(x_steps[:max_x], med_val, color=cols[sub], linewidth=1, linestyle='dashed')
            axs[0].fill_between(x_steps[:max_x], mad_lo, mad_up, facecolor=cols[sub], alpha=0.2)
            axs[0].plot(x_steps[:max_x], val, color=cols[sub], linewidth=1, label='label')
            
            axs[0].grid(True, linewidth=0.2)
    
    axs[0].set_ylabel('Valuation (log10)', size='small')
    axs[0].text(0.5, -0.3, "(a)", size='small', transform=axs[0].transAxes)
    axs[0].set_xlabel('Steps', size='small')

    for sub in range(3):
            lev = inv2_data[0, 0, 12 + sub, :max_x].reshape(-1)
            lmad = inv2_data[0, 0, 15 + sub, :max_x].reshape(-1)
            lmad_up = (lev + lmad).reshape(-1)
            lmad_lo = np.maximum(0, lev - lmad).reshape(-1)
            med_lev = inv2_data[0, 0, 21 + sub, :max_x].reshape(-1)
            
            axs[1].plot(x_steps[:max_x], med_lev, color=cols[sub], linewidth=1, linestyle='dashed')
            axs[1].fill_between(x_steps[:max_x], lmad_lo, lmad_up, facecolor=cols[sub], alpha=0.2)
            axs[1].plot(x_steps[:max_x], lev, color=cols[sub], linewidth=1)

            axs[1].grid(True, linewidth=0.2)

    axs[1].set_ylabel('Leverage', size='small')
    axs[1].text(0.5, -0.3, "(b)", size='small', transform=axs[1].transAxes)
    fig.legend(handles=[avg_col, top_col, adj_col], loc='upper center', ncol=3, frameon=False, fontsize='medium')

    axs[0].tick_params(axis='both', which='major', labelsize='small')
    axs[1].tick_params(axis='both', which='major', labelsize='small')

    fig.subplots_adjust(bottom=0.25)

    plt.savefig(filename_png, dpi=400, format='png')

def plot_inv3(inv3_data: np.ndarray, filename_png: str):
    """
    Investor 3 density plot.

    Parameters:
        inv3_data: array of investor 3 data across all stop-losses and retention ratios
        filename_png (directory): save path of plot
    """
    roll = inv3_data[:, 0, -1, 0]
    stop = inv3_data[0, :, -2, 0]

    r_len = roll.shape[0]
    s_len = stop.shape[0]
    mean = np.zeros((r_len, s_len))
    adj_mean = np.zeros((r_len, s_len))
    med = np.zeros((r_len, s_len))
    adj_med = np.zeros((r_len, s_len))

    for r in range(r_len):
        for s in range(s_len):
            mean[r, s] = inv3_data[r, s, 0, -1]
            adj_mean[r, s] = inv3_data[r, s, 2, -1]
            med[r, s] = inv3_data[r, s, 9, -1]
            adj_med[r, s] = inv3_data[r, s, 11, -1] 

    mean, med = np.log10(mean), np.log10(med)
    adj_mean, adj_med = np.log10(adj_mean), np.log10(adj_med)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    
    im0 = axs[0, 0].pcolormesh(stop*100, roll*100, mean, shading='gouraud', vmin=adj_mean.min(), vmax=mean.max())
    im1 = axs[0, 1].pcolormesh(stop*100, roll*100, adj_mean, shading='gouraud', vmin=adj_mean.min(), vmax=mean.max())
    cb1 = fig.colorbar(im1, ax=axs[0, 1])

    im2 = axs[1, 0].pcolormesh(stop*100, roll*100, med, shading='gouraud', vmin=adj_med.min(), vmax=med.max())
    im3 = axs[1, 1].pcolormesh(stop*100, roll*100, adj_med, shading='gouraud', vmin=adj_med.min(), vmax=med.max())
    cb2 = fig.colorbar(im3, ax=axs[1, 1])

    axs[0, 0].tick_params(axis='both', which='major', labelsize='small')
    axs[0, 0].set_title('Mean (log10)', size='medium')
    axs[0, 0].set_ylabel('Retention Ratio Φ (%)', size='small')
    axs[0, 0].set_xlabel('Stop-Loss λ (%)', size='small')
    axs[0, 0].text(0.5, -0.25, "(a)", size='small', transform=axs[0, 0].transAxes)

    axs[0, 1].tick_params(axis='both', which='major', labelsize='small')
    axs[0, 1].set_ylabel('', size='small')
    axs[0, 1].set_xlabel('', size='small')
    axs[0, 1].set_title('Adjusted Mean (log10)', size='medium')
    axs[0, 1].text(0.5, -0.25, "(b)", size='small', transform=axs[0, 1].transAxes)
    cb1.ax.tick_params(labelsize='small')

    axs[1, 0].tick_params(axis='both', which='major', labelsize='small')
    axs[1, 0].set_ylabel('Retention Ratio Φ (%)', size='small')
    axs[1, 0].set_xlabel('Stop-Loss λ (%)', size='small')
    axs[1, 0].set_title('Median (log10)', size='medium')
    axs[1, 0].text(0.5, -0.25, "(a)", size='small', transform=axs[1, 0].transAxes)

    axs[1, 1].tick_params(axis='both', which='major', labelsize='small')
    axs[1, 1].set_ylabel('', size='small')
    axs[1, 1].set_xlabel('', size='small')
    axs[1, 1].set_title('Adjusted Median (log10)', size='medium')
    axs[1, 1].text(0.5, -0.25, "(b)", size='small', transform=axs[1, 1].transAxes)
    cb2.ax.tick_params(labelsize='small')

    fig.subplots_adjust(hspace=0.4)

    plt.savefig(filename_png, dpi=400, format='png')

def plot_inv4(inv4_data: np.ndarray, filename_png: str):
    """
    Investor 4 density plots for only three probabilities.

    Parameters:
        inv4_data: array of investor 4 data across various returns and three probabilities.
        filename_png (directory): save path of plot
    """
    pu_len, ru_len, rd_len, _ = inv4_data.shape

    pu = inv4_data[:, 0, 0, 0]
    ru = inv4_data[0, :, 0, 1]
    rd = inv4_data[0, 0, :, 2]

    kelly = np.zeros((pu_len, ru_len, rd_len))

    for q in range(pu_len):
        for r in range(ru_len):
            for s in range(rd_len):
                kelly[q, r, s] = inv4_data[q, r, s, 3] 

    fig, axs = plt.subplots(nrows=1, ncols=int(pu_len), figsize=(10, 3))

    im0 = axs[0].pcolormesh(rd * 100, ru * 100, kelly[0], shading='gouraud', vmin=kelly[0].min(), vmax=kelly[0].max())
    cb0 = fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].pcolormesh(rd * 100, ru * 100, kelly[1], shading='gouraud', vmin=kelly[1].min(), vmax=kelly[1].max())
    cb1 = fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].pcolormesh(rd * 100, ru * 100, kelly[2], shading='gouraud', vmin=kelly[2].min(), vmax=kelly[2].max())
    cb2 = fig.colorbar(im2, ax=axs[2])

    axs[0].tick_params(axis='both', which='major', labelsize='small')
    axs[0].set_ylabel('Up Return (%)', size='small')
    axs[0].set_xlabel('Down Return (%)', size='small')
    axs[0].set_title(str(int(pu[0] * 100))+'% Up Probability', size='medium')
    axs[0].text(0.5, -0.325, "(a)", size='small', transform=axs[0].transAxes)
    cb0.ax.tick_params(labelsize='small')

    axs[1].tick_params(axis='both', which='major', labelsize='small')
    axs[1].set_ylabel('', size='small')
    axs[1].set_xlabel('', size='small')
    axs[1].set_title(str(int(pu[1] * 100))+'% Up Probability', size='medium')
    axs[1].text(2, -0.325, "(b)", size='small', transform=axs[0].transAxes)
    cb1.ax.tick_params(labelsize='small')

    axs[2].tick_params(axis='both', which='major', labelsize='small')
    axs[2].set_ylabel('', size='small')
    axs[2].set_xlabel('', size='small')
    axs[2].set_title(str(int(pu[2] * 100))+'% Up Probability', size='medium')
    axs[2].text(3.5, -0.325, "(c)", size='small', transform=axs[0].transAxes)
    cb2.ax.tick_params(labelsize='small')

    fig.subplots_adjust(bottom=0.25)
    
    plt.savefig(filename_png, dpi=400, format='png')