import extras.utils as utils
import matplotlib.pyplot as plt
import numpy as np

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

    exp = utils.get_exponent(cum_steps)
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
    
    exp = utils.get_exponent(cum_steps)
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

    eval_exp = utils.get_exponent(input_dict['eval_freq'])
    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * int(input_dict['n_eval']))
    scores = np.zeros((count_x, count_y))
    max_score = np.zeros((count_x, count_y))

    for t in range(count_x):
        for n in range(input_dict['n_trials']):
            for s in range(int(input_dict['n_eval'])):
                scores[t, s + n * int(input_dict['n_eval'])] = eval_log[n, t, s, 1]
                max_score[t, s + n * int(input_dict['n_eval'])] = int(input_dict['max_eval_reward'])

    score_limit = np.mean(max_score, axis=1, keepdims=True)
    score_mean = np.mean(scores, axis=1, keepdims=True)
    score_max, score_min = np.max(scores, axis=1, keepdims=True), np.min(scores, axis=1, keepdims=True)

    score_mad = np.mean(np.abs(scores - score_mean), axis=1, keepdims=True)
    score_mad_up = np.minimum(score_max, score_mean+score_mad, score_limit).reshape(-1)
    score_mad_lo = np.maximum(score_min, score_mean-score_mad).reshape(-1)

    score_mean = score_mean.reshape(-1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, label='score')

    ax1.plot(x_steps, score_mean, color='C0')
    ax1.fill_between(x_steps, score_mad_lo, score_mad_up, facecolor='C0', alpha=0.4)
    ax1.set_xlabel('Steps (1e'+str(exp)+')')
    ax1.yaxis.tick_left()
    ax1.set_ylabel('Mean Score')
    ax1.yaxis.set_label_position('left')
    ax1.grid(True, linewidth=0.5)

    tit1 = 'Mean and MAD of '+str(input_dict['n_trials'])+'x'+str(int(input_dict['n_eval']))+' Evaluations per '
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

    eval_exp = utils.get_exponent(input_dict['eval_freq'])
    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * int(input_dict['n_eval']))

    scores = np.zeros((count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((count_x, count_y))
    minl = np.zeros((count_x, count_y))
    maxl = np.zeros((count_x, count_y))
    shadow = np.zeros((count_x, count_y))

    for t in range(count_x):
        for n in range(input_dict['n_trials']):
            for s in range(int(input_dict['n_eval'])):
                scores[t, s + n * int(input_dict['n_eval'])] = eval_log[n, t, s, 1]
                loss[t, s + n * int(input_dict['n_eval'])] = np.mean(eval_log[n, t, s, 3:5])
                # minl[t, s + n * int(input_dict['n_eval'])] = np.mean(eval_log[n, t, s, 5:7])
                # maxl[t, s + n * int(input_dict['n_eval'])] = np.mean(eval_log[n, t, s, 7:9])
                # shadow[t, s + n * int(input_dict['n_eval'])] = np.mean(eval_log[n, t, s, 9:11])

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
    loss_mean = loss_mean.reshape(-1)

    # minl_mean = np.mean(minl, axis=1, keepdims=True)
    # minl_max, minl_min = np.max(minl, axis=1, keepdims=True), np.min(minl, axis=1, keepdims=True)
    # minl_mad = np.mean(np.abs(minl - minl_mean), axis=1, keepdims=True)
    # minl_mad_up = np.minimum(minl_max, minl_mean+minl_mad).reshape(-1)
    # minl_mad_lo = np.maximum(minl_min, minl_mean-minl_mad).reshape(-1)
    # minl_mean = minl_mean.reshape(-1)

    # maxl_mean = np.mean(maxl, axis=1, keepdims=True)
    # maxl_max, maxl_min = np.max(maxl, axis=1, keepdims=True), np.min(maxl, axis=1, keepdims=True)
    # maxl_mad = np.mean(np.abs(maxl - maxl_mean), axis=1, keepdims=True)
    # maxl_mad_up = np.minimum(maxl_max, maxl_mean+maxl_mad).reshape(-1)
    # maxl_mad_lo = np.maximum(maxl_min, maxl_mean-maxl_mad).reshape(-1)
    # maxl_mean = maxl_mean.reshape(-1)

    # shadow_mean = np.mean(shadow, axis=1, keepdims=True)
    # shadow_max, shadow_min = np.max(shadow, axis=1, keepdims=True), np.min(shadow, axis=1, keepdims=True)
    # shadow_mad = np.mean(np.abs(shadow - shadow_mean), axis=1, keepdims=True)
    # shadow_mad_up = np.minimum(shadow_max, shadow_mean+shadow_mad).reshape(-1)
    # shadow_mad_lo = np.maximum(shadow_min, shadow_mean-shadow_mad).reshape(-1)
    # shadow_mean = shadow_mean.reshape(-1)

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
    # ax2.plot(x_steps, minl_mean, color='C4')
    # ax2.fill_between(x_steps, minl_mad_lo, minl_mad_up, facecolor='C4', alpha=0.4)
    # ax2.plot(x_steps, maxl_mean, color='C5')
    # ax2.fill_between(x_steps, maxl_mad_lo, maxl_mad_up, facecolor='C5', alpha=0.4)
    # ax2.plot(x_steps, shadow_mean, color='C6')
    # ax2.fill_between(x_steps, shadow_mad_lo, shadow_mad_up, facecolor='C6', alpha=0.4)

    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Critic Loss', color='C3')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C3')

    ymin, ymax = ax2.get_ylim()
    ax2.set(ylim=(ymin, ymax))

    tit1 = 'Mean, MAD, and STD of '+str(input_dict['n_trials'])+'x'+str(int(input_dict['n_eval']))+' Evaluations per '
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

    eval_exp = utils.get_exponent(input_dict['eval_freq'])
    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log/10**(exp)

    count_x = int(input_dict['n_cumsteps'] / input_dict['eval_freq'])
    count_y = int(input_dict['n_trials'] * int(input_dict['n_eval']))

    scores = np.zeros((count_x, count_y))
    max_score = np.ones((count_x, count_y)) * int(input_dict['max_eval_reward'])
    loss = np.zeros((count_x, count_y))

    for t in range(count_x):
        for n in range(input_dict['n_trials']):
            for s in range(int(input_dict['n_eval'])):
                scores[t, s + n * int(input_dict['n_eval'])] = eval_log[n, t, s, 1]
                loss[t, s + n * int(input_dict['n_eval'])] = np.mean(eval_log[n, t, s, 3:5])

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

    tit1 = 'Mean and MAD of '+str(input_dict['n_trials'])+'x'+str(int(input_dict['n_eval']))+' Evaluations per '
    tit2 = str(int(input_dict['eval_freq']))[0]+'e'+str(eval_exp)+' Steps \n'
    tit3 = input_dict['algo']+'-'+input_dict['s_dist']+': \''+input_dict['env_id']+' ('+'d'+input_dict['dynamics']+', '
    tit4 = input_dict['loss_fn']+', '+'b'+str(input_dict['buffer']/1e6)[0]+', '+'m'+str(input_dict['multi_steps'])+')'
    title = tit1 + tit2 + tit3 + tit4

    ax1.set_title(title)
    
    plt.savefig(filename_png, dpi=600, format='png')