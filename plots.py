import matplotlib.pyplot as plt
import numpy as np
import torch as T
from typing import List
import utils

def get_exponent(array: np.ndarray) -> int:
    """
    Obtain expoenent for maximum array value used for scaling and axis labels.

    Parameters:
        array: array of usually cumulative steps in trial

    Returns:
        exp: exponent of max cumulative steps
    """
    max_step = np.max(array)

    if str(max_step)[0] == 1:
        exp = int(len(str(int(max_step))))
    else:
        exp = int(len(str(int(max_step))) - 1)

    return exp

def plot_learning_curve(input_dict: dict, trial_log: np.ndarray, filename_png: str):
    """
    Plot of game running average score and critic loss for environment.
    
    Parameters:
        input_dict: dictionary containing all execution details
        trial_log: log of episode data of a single trial
        filename_png (directory): save path of plot
    """
    # truncate log up to maximum episodes
    try:
        trial_log = trial_log[:np.min(np.where(trial_log[:, 0] == 0))]
    except:
        pass
    
    score_log = trial_log[:, 1]
    steps = trial_log[:, 2]
    critic_log = trial_log[:, 3:5].sum(axis=1)

    # ignore intial NaN critic loss when batch_size > buffer
    idx, loss = 0, 0
    while np.nan_to_num(loss) == 0:
        loss = critic_log[idx]
        idx += 1

    offset = np.max(idx - 1, 0)
    score_log = score_log[offset:]
    steps = steps[offset:]
    critic_log = critic_log[offset:]
    length = len(score_log)

    # obtain cumulative steps for x-axis
    cum_steps = np.zeros(length)
    cum_steps[0] = steps[0]
    for i in range(length-1):
        cum_steps[i+1] = steps[i+1] + cum_steps[i]

    exp = get_exponent(cum_steps)
    x_steps = cum_steps/10**(exp)
    
    # calculate moving averages
    trail = input_dict['trail']
    running_avg1 = np.zeros(length)
    for i in range(length-offset):
        running_avg1[i+offset] = np.mean(score_log[max(0, i-trail):(i+1)])

    running_avg2 = np.zeros(length)
    for i in range(length-offset):
        running_avg2[i+offset] = np.mean(critic_log[max(0, i-trail):(i+1)])

    warmup_end_idx = np.min(np.where(np.array(x_steps) - input_dict['random']/10**(exp) > 0))
    running_avg2[:warmup_end_idx] = [0 for x in range(warmup_end_idx)]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='score')
    ax2 = fig.add_subplot(1,1,1, label='critic', frame_on=False)

    ax1.plot(x_steps, running_avg1, color='C0')
    ax1.set_xlabel('Training Steps (1e'+str(exp)+')')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Average Score', color='C0')
    ax1.yaxis.set_label_position('left')
    ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, linewidth=0.5)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    ax2.plot(x_steps, running_avg2, color='C3')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Average Critic Loss', color='C3')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C3')
    
    # make vertical lines splitting fractions of total episodes
    partitions = 0.25
    for block in range(int(1/partitions-1)):
        period = x_steps[int(np.min(length * partitions * (block + 1)))-1]
        ax1.vlines(x=period, ymin=ymin, ymax=ymax, linestyles ="dashed", color='C2')

    # make vertical line when algorithm begins learning
    # ax1.vlines(x=x_steps[warmup_end_idx], ymin=ymin, ymax=ymax, linestyles ="dashed", color='C7')

    tit1 = 'Trailing '+str(int(input_dict['trail']))+' Episode Averages and '+str(partitions)[2:4]+'% Partitions \n'
    tit2 = input_dict['algo']+'-'+input_dict['s_dist']+': \''+input_dict['env_id']+'\' '+'('+'d'+input_dict['dynamics']+', '
    tit3 = input_dict['loss_fn']+', '+'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+', '
    tit4 = 'e'+str(int(length))+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)

    plt.savefig(filename_png, bbox_inches='tight', dpi=600, format='png')

def plot_trial_curve(input_dict: dict, trial_log: np.ndarray, filename_png: str):
    """
    Plot of interpolated mean, MAD, and STD score and critic loss across all trials for environment.
    
    Parameters:
        input_dict: dictionary containing all execution details
        trial_log: log of episode data
        filename_png (directory): save path of plot
    """
    score_log = trial_log[:, :, 1]
    steps_log = trial_log[:, :, 2]
    critic_log = trial_log[:, :, 3:5].sum(axis=2)

    # find maximum number of episodes in each trial
    max_episodes = []
    for trial in range(steps_log.shape[0]):
        try:
            max_episodes.append(np.min(np.where(steps_log[trial, :] == 0)))
        except:
            max_episodes.append(steps_log.shape[1])

    # ignore intial NaN critic loss when batch_size > buffer
    offset = []
    for trial in range(steps_log.shape[0]):
        idx, loss = 0, 0

        while np.nan_to_num(loss) == 0:
            loss = critic_log[trial, idx]
            idx += 1

        offset.append(idx)
    
    max_offset = np.maximum(np.array(offset) - 1, 0)
    small_max_offset = np.min(max_offset)
    length = steps_log.shape[1] - small_max_offset 

    scores = np.zeros((steps_log.shape[0], length))
    steps = np.zeros((steps_log.shape[0], length))
    critics = np.zeros((steps_log.shape[0], length))

    for trial in range(steps.shape[0]):
        scores[trial, :length + small_max_offset - max_offset[trial]] = score_log[trial, max_offset[trial]:]
        steps[trial, :length + small_max_offset - max_offset[trial]] = steps_log[trial, max_offset[trial]:]
        critics[trial, :length + small_max_offset - max_offset[trial]] = critic_log[trial, max_offset[trial]:]

    # obtain cumulative steps for x-axis for each trial
    cum_steps = np.zeros((steps.shape[0], length))
    cum_steps[:, 0] = steps[:, 0]
    for trial in range(steps.shape[0]):
        for e in range(max_episodes[trial]-max_offset[trial]-1):
            cum_steps[trial, e+1] = steps[trial, e+1] + cum_steps[trial, e]
    
    exp = get_exponent(cum_steps)
    x_steps = cum_steps/10**(exp)   

    # create lists for interteploation
    list_steps, list_scores, list_critic = [], [], []
    for trial in range(scores.shape[0]):
        trial_step, trial_score, trial_critic = [], [], []
        for epis in range(max_episodes[trial]-max_offset[trial]):
            trial_step.append(x_steps[trial, epis])
            trial_score.append(scores[trial, epis])
            trial_critic.append(critics[trial, epis])
        list_steps.append(trial_step)
        list_scores.append(trial_score)
        list_critic.append(trial_critic)

    # linearly interpolate mean, MAD and STD across trials
    count_x = list_steps[max_episodes.index(max(max_episodes))]
    score_interp = [np.interp(count_x, list_steps[i], list_scores[i]) for i in range(steps.shape[0])]
    critic_interp = [np.interp(count_x, list_steps[i], list_critic[i]) for i in range(steps.shape[0])]

    score_mean = np.mean(score_interp, axis=0)
    score_max, score_min = np.max(score_interp, axis=0), np.min(score_interp, axis=0)
    score_mad = np.mean(np.abs(score_interp - score_mean), axis=0)
    score_mad_up, score_mad_lo = np.minimum(score_max, score_mean+score_mad), np.maximum(score_min, score_mean-score_mad)
    score_std = np.std(score_interp, ddof=0, axis=0)
    score_std_up, score_std_lo = np.minimum(score_max, score_mean+score_std), np.maximum(score_min, score_mean-score_std)

    critic_mean = np.mean(critic_interp, axis=0)
    critic_max, critic_min = np.max(critic_interp, axis=0), np.min(critic_interp, axis=0)
    critic_mad = np.mean(np.abs(critic_interp - critic_mean), axis=0)
    critic_mad_up, critic_mad_lo = np.minimum(critic_max, critic_mean+critic_mad), np.maximum(critic_min, critic_mean-critic_mad)
    critic_std = np.std(critic_interp, ddof=0, axis=0)
    critic_std_up, critic_std_lo = np.minimum(critic_max, critic_mean+critic_std), np.maximum(critic_min, critic_mean-critic_std)

    warmup_end_idx = np.min(np.where(np.array(count_x) - input_dict['random']/10**(exp) > 0))
    critic_mean[:warmup_end_idx] = [0 for x in range(warmup_end_idx)]
    critic_mad_up[:warmup_end_idx] = [0 for x in range(warmup_end_idx)]
    critic_mad_lo[:warmup_end_idx] = [0 for x in range(warmup_end_idx)]
    critic_std_up[:warmup_end_idx] = [0 for x in range(warmup_end_idx)]
    critic_std_lo[:warmup_end_idx] = [0 for x in range(warmup_end_idx)]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='score')
    ax2 = fig.add_subplot(1,1,1, label='critic', frame_on=False)

    ax1.plot(count_x, score_mean, color='C0', linewidth=0.8)
    ax1.fill_between(count_x, score_mad_lo, score_mad_up, facecolor='C0', alpha=0.6)
    ax1.fill_between(count_x, score_std_lo, score_std_up, facecolor='C0', alpha=0.2)
    ax1.set_xlabel('Training Steps (1e'+str(exp)+')')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Score', color='C0')
    ax1.yaxis.set_label_position('left')
    ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, linewidth=0.5)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    ax2.plot(count_x, critic_mean, color='C3', linewidth=0.8)
    ax2.fill_between(count_x, critic_mad_lo, critic_mad_up, facecolor='C3', alpha=0.6)
    ax2.fill_between(count_x, critic_std_lo, critic_std_up, facecolor='C3', alpha=0.2)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Critic Loss', color='C3')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C3')

    tit1 = 'Linearly Interpolated Mean, MAD, and STD of '+str(input_dict['n_trials'])+' Trials \n'
    tit2 = input_dict['algo']+'-'+input_dict['s_dist']+': \''+input_dict['env_id']+'\' '+'('+'d'+input_dict['dynamics']+', '
    tit3 = input_dict['loss_fn']+', '+'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+', '
    tit4 = 'e'+str(int(length))+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)
    
    plt.savefig(filename_png, dpi=600, format='png')

def plot_eval_curve(input_dict: dict, eval_log: np.ndarray, filename_png: str):
    """
    Plot of mean, MAD and STD scores of evaluation episodes for all trials in environment.
    
    Parameters:
        input_dict: dictionary containing all execution details
        eval_log: log of episode data for all trials
        filename_png (directory): save path of plot
    """
    cum_steps_log = eval_log[0, :, 0, -1]

    eval_exp = get_exponent(input_dict['eval_freq'])
    exp = get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * input_dict['n_eval'])
    scores = np.zeros((count_x, count_y))
    max_score = np.zeros((count_x, count_y))

    for t in range(count_x):
        for n in range(input_dict['n_trials']):
            for s in range(input_dict['n_eval']):
                scores[t, s + n * input_dict['n_eval']] = eval_log[n, t, s, 1]
                max_score[t, s + n * input_dict['n_eval']] = int(input_dict['max_eval_reward'])

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    score_mean = np.mean(scores, axis=1, keepdims=True)
    score_max, score_min = np.max(scores, axis=1, keepdims=True), np.min(scores, axis=1, keepdims=True)

    score_mad = np.mean(np.abs(scores - score_mean), axis=1, keepdims=True)
    score_mad_up = np.minimum(score_max, score_mean+score_mad, score_limit).reshape(-1)
    score_mad_lo = np.maximum(score_min, score_mean-score_mad).reshape(-1)
    score_std = np.std(scores, axis=1, ddof=0, keepdims=True)
    score_std_up = np.minimum(score_max, score_mean+score_std, score_limit).reshape(-1)
    score_std_lo = np.maximum(score_min, score_mean-score_std).reshape(-1)

    score_mean = score_mean.reshape(-1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='score')

    ax1.plot(x_steps, score_mean, color='C0')
    ax1.fill_between(x_steps, score_mad_lo, score_mad_up, facecolor='C0', alpha=0.4)
    ax1.fill_between(x_steps, score_std_lo, score_std_up, facecolor='C0', alpha=0.2)
    ax1.set_xlabel('Steps (1e'+str(exp)+')')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Mean Score')
    ax1.yaxis.set_label_position('left')
    ax1.grid(True, linewidth=0.5)

    tit1 = 'Mean, MAD, and STD of '+str(input_dict['n_trials'])+'x'+str(input_dict['n_eval'])+' Evaluations per '
    tit2 = str(int(input_dict['eval_freq']))[0]+'e'+str(eval_exp)+' Steps \n'
    tit3 = input_dict['algo']+'-'+input_dict['s_dist']+': \''+input_dict['env_id']+'\' '+'('+'d'+input_dict['dynamics']+', '
    tit4 = input_dict['loss_fn']+', '+'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)
    
    plt.savefig(filename_png, dpi=600, format='png')

def plot_eval_loss_2d(input_dict: dict, eval_log: np.ndarray, filename_png: str):
    """
    2D plot of Mean, MAD, and STD of scores and twin critic loss during evaluation episodes for all trials in environment.
    
    Parameters:
        input_dict: 
        eval_log: log of episode data for all trials
        filename_png (directory): save path of plot
    """
    cum_steps_log = eval_log[0, :, 0, -1]

    eval_exp = get_exponent(input_dict['eval_freq'])
    exp = get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * input_dict['n_eval'])

    scores = np.zeros((count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((count_x, count_y))

    for t in range(count_x):
        for n in range(input_dict['n_trials']):
            for s in range(input_dict['n_eval']):
                scores[t, s + n * input_dict['n_eval']] = eval_log[n, t, s, 1]
                loss[t, s + n * input_dict['n_eval']] = np.mean(eval_log[n, t, s, 3:5])

    score_limit = np.mean(max_score, axis=1, keepdims=True)

    score_mean = np.mean(scores, axis=1, keepdims=True)
    score_max, score_min = np.max(scores, axis=1, keepdims=True), np.min(scores, axis=1, keepdims=True)
    score_mad = np.mean(np.abs(scores - score_mean), axis=1, keepdims=True)
    score_mad_up = np.minimum(score_max, score_mean+score_mad, score_limit).reshape(-1)
    score_mad_lo = np.maximum(score_min, score_mean-score_mad).reshape(-1)
    score_std = np.std(scores, axis=1, ddof=0, keepdims=True)
    score_std_up = np.minimum(score_max, score_mean+score_std, score_limit).reshape(-1)
    score_std_lo = np.maximum(score_min, score_mean-score_std).reshape(-1)
    score_mean = score_mean.reshape(-1)

    loss_mean = np.mean(loss, axis=1, keepdims=True)
    loss_max, loss_min = np.max(loss, axis=1, keepdims=True), np.min(loss, axis=1, keepdims=True)
    loss_mad = np.mean(np.abs(loss - loss_mean), axis=1, keepdims=True)
    loss_mad_up = np.minimum(loss_max, loss_mean+loss_mad).reshape(-1)
    loss_mad_lo = np.maximum(loss_min, loss_mean-loss_mad).reshape(-1)
    loss_std = np.std(loss, axis=1, ddof=0, keepdims=True)
    loss_std_up = np.minimum(loss_max, loss_mean+loss_std).reshape(-1)
    loss_std_lo = np.maximum(loss_min, loss_mean-loss_std).reshape(-1)
    loss_mean = loss_mean.reshape(-1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='score')
    ax2 = fig.add_subplot(1,1,1, label='loss', frame_on=False)

    ax1.plot(x_steps, score_mean, color='C0')
    ax1.fill_between(x_steps, score_mad_lo, score_mad_up, facecolor='C0', alpha=0.4)
    ax1.fill_between(x_steps, score_std_lo, score_std_up, facecolor='C0', alpha=0.2)
    ax1.set_xlabel('Steps (1e'+str(exp)+')')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Mean Score',  color='C0')
    ax1.yaxis.set_label_position('left')
    ax1.tick_params(axis='y', colors='C0')
    ax1.grid(True, linewidth=0.5)

    ax2.plot(x_steps, loss_mean, color='C3')
    ax2.fill_between(x_steps, loss_mad_lo, loss_mad_up, facecolor='C3', alpha=0.4)
    ax2.fill_between(x_steps, loss_std_lo, loss_std_up, facecolor='C3', alpha=0.2)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Critic Loss', color='C3')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C3')

    tit1 = 'Mean, MAD, and STD of '+str(input_dict['n_trials'])+'x'+str(input_dict['n_eval'])+' Evaluations per '
    tit2 = str(int(input_dict['eval_freq']))[0]+'e'+str(eval_exp)+' Steps \n'
    tit3 = input_dict['algo']+'-'+input_dict['s_dist']+': \''+input_dict['env_id']+' ('+'d'+input_dict['dynamics']+', '
    tit4 = input_dict['loss_fn']+', '+'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)
    
    plt.savefig(filename_png, dpi=600, format='png')

def plot_eval_loss_3d(input_dict: dict, eval_log: np.ndarray, filename_png: str):
    """
    3D plot of Mean and MAD of scores and twin critic loss during evaluation episodes for all trials in environment.
    
    Parameters:
        input_dict: dictionary containing all execution details
        eval_log: log of episode data for all trials
        filename_png (directory): save path of plot
    """
    cum_steps_log = eval_log[0, :, 0, -1]

    eval_exp = get_exponent(input_dict['eval_freq'])
    exp = get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * input_dict['n_eval'])

    scores = np.zeros((count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((count_x, count_y))

    for t in range(count_x):
        for n in range(input_dict['n_trials']):
            for s in range(input_dict['n_eval']):
                scores[t, s + n * input_dict['n_eval']] = eval_log[n, t, s, 1]
                loss[t, s + n * input_dict['n_eval']] = np.mean(eval_log[n, t, s, 3:5])

    score_limit = np.mean(max_score, axis=1, keepdims=True)

    score_mean = np.mean(scores, axis=1, keepdims=True)
    score_max, score_min = np.max(scores, axis=1, keepdims=True), np.min(scores, axis=1, keepdims=True)
    score_mad = np.mean(np.abs(scores - score_mean), axis=1, keepdims=True)
    score_mad_up = np.minimum(score_max, score_mean+score_mad, score_limit).reshape(-1)
    score_mad_lo = np.maximum(score_min, score_mean-score_mad).reshape(-1)
    score_mean = score_mean.reshape(-1)
                
    loss_mean = np.mean(loss, axis=1, keepdims=True)
    loss_max, loss_min = np.max(loss, axis=1, keepdims=True), np.min(loss, axis=1, keepdims=True)
    loss_mad = np.mean(np.abs(loss - loss_mean), axis=1, keepdims=True)
    loss_mad_up = np.minimum(loss_max, loss_mean+loss_mad).reshape(-1)
    loss_mad_lo = np.maximum(loss_min, loss_mean-loss_mad).reshape(-1)
    loss_mean = loss_mean.reshape(-1)

    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    ax1.plot3D(x_steps, score_mean, loss_mean, color='k')

    for i in range(len(x_steps)):
        ax1.plot([x_steps[i], x_steps[i]], [score_mad_up[i], score_mad_lo[i]], [loss_mean[i], loss_mean[i]], 
                color='C0', alpha=0.5, linewidth=1)
        ax1.plot([x_steps[i], x_steps[i]], [score_mean[i], score_mean[i]], [loss_mad_up[i], loss_mad_lo[i]], 
                color='C3', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Steps (1e'+str(exp)+')')
    ax1.set_ylabel('Mean Score')
    ax1.set_zlabel('Critic Loss')

    tit1 = 'Mean and MAD of '+str(input_dict['n_trials'])+'x'+str(input_dict['n_eval'])+' Evaluations per '
    tit2 = str(int(input_dict['eval_freq']))[0]+'e'+str(eval_exp)+' Steps \n'
    tit3 = input_dict['algo']+'-'+input_dict['s_dist']+': \''+input_dict['env_id']+' ('+'d'+input_dict['dynamics']+', '
    tit4 = input_dict['loss_fn']+', '+'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)
    
    plt.savefig(filename_png, dpi=600, format='png')

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

    exp = get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * input_dict['n_eval'])

    scores = np.zeros((algos, closs, count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((algos, closs, count_x, count_y)) 

    for a in range(algos):
        for l in range(closs):
            for t in range(count_x):
                for n in range(input_dict['n_trials']):
                    for s in range(input_dict['n_eval']):
                        scores[a, l, t, s + n * input_dict['n_eval']] = data[a, l, n, t, s, 1]
                        loss[a, l, t, s + n * input_dict['n_eval']] = np.mean(data[a, l, n, t, s, 3:5])

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

    plt.savefig(filename_png, dpi=600, format='png')

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

    exp = get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * input_dict['n_eval'])
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

                    for s in range(input_dict['n_eval']):
                        scores[a, l, t, s + n * input_dict['n_eval']] = data[a, l, n, t, s, 1]

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

    plt.savefig(filename_png, dpi=600, format='png')

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

    exp = get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * input_dict['n_eval'])
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

                    # for s in range(input_dict['n_eval']):
                    #     scores[a, l, t, s + n * input_dict['n_eval']] = data[a, l, n, t, s, 1]

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

    plt.savefig(filename_png, dpi=600, format='png')

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

    exp = get_exponent(cum_steps_log)
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

    plt.savefig(filename_png, dpi=600, format='png')

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

def loss_fn_plot(filename_png: str):
    """
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

    plt.savefig(filename_png, dpi=1000, format='png')