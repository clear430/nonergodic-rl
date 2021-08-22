from datetime import datetime
import gym
import numpy as np
import pybullet_envs
import time
import torch as T
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
from typing import Tuple

def eval_policy(agent: object, inputs: dict, eval_log: np.ndarray, cum_steps: int, round: int, 
                eval_run: int, loss: Tuple[float, float, float, float, float, float], logtemp: float, 
                loss_params: Tuple[float, float, float, float]):
    """
    Evaluates agent policy on environment without learning for a fixed number of episodes.

    Parameters:
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of exiting evalaution results
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    print('{} {}-{}-{}-{} {} Evaluations cst {}:'.format(datetime.now().strftime('%d %H:%M:%S'), 
    inputs['algo'], inputs['s_dist'], inputs['loss_fn'], round+1, inputs['n_eval'], cum_steps))

    print('{} Training Summary: T/Cg/Cs {:1.2f}/{:1.2f}/{:1.2f}, C/A {:1.1f}/{:1.1f}'
    .format(datetime.now().strftime('%d %H:%M:%S'), np.exp(logtemp), sum(loss_params[0:2])/2, 
        sum(loss_params[2:4])/2, sum(loss[0:2]), loss[2]))
    
    eval_env = gym.make(inputs['env_id'])
    
    for eval in range(int(inputs['n_eval'])):
        start_time = time.perf_counter()
        run_state = eval_env.reset()
        run_done = False
        run_step, run_reward = 0, 0

        while not run_done:
            run_action, _ = agent.select_next_action(run_state)
            run_next_state, eval_reward, run_done, _ = eval_env.step(run_action)
            run_reward += eval_reward
            run_state = run_next_state
            run_step += 1

            # prevent evaluation from running forever
            if run_reward >= int(inputs['max_eval_reward']):
                    break
        
        end_time = time.perf_counter()
        
        eval_log[round, eval_run, eval, 0] = end_time - start_time
        eval_log[round, eval_run, eval, 1] = run_reward
        eval_log[round, eval_run, eval, 2] = run_step
        eval_log[round, eval_run, eval, 3:14] = loss
        eval_log[round, eval_run, eval, 14] = logtemp
        eval_log[round, eval_run, eval, 15:19] = loss_params
        eval_log[round, eval_run, eval, 19] = cum_steps
    
        print('{} Episode {}: r/st {:1.0f}/{}'
        .format(datetime.now().strftime('%d %H:%M:%S'), eval, run_reward, run_step))

    run = eval_log[round, eval_run, :, 1]
    mean_run = np.mean(run)
    mad_run = np.mean(np.abs(run - mean_run))
    std_run = np.std(run, ddof=0)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    stats = [mean_run, mean_step, mad_run, mad_step, std_run, std_step]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(eval_log[round, eval_run, :, 3])

    print("{} Evaluations Summary {:1.0f}/s r/st: mean {:1.0f}/{:1.0f}, mad {:1.0f}/{:1.0f}, std {:1.0f}/{:1.0f}"
    .format(datetime.now().strftime('%d %H:%M:%S'), steps_sec, 
            stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]))

def save_directory(inputs: dict, round: int) -> str:
    """
    Provides string directory for data and plot saving names.

    Parameters:
        inputs: dictionary containg all execution details
        round: current round of trials

    Returns:
        directory: file path and name to give to current experiment plots
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    dir1 = 'results/'+inputs['env_id']+'/'
    dir2 = inputs['env_id']+'--'+inputs['algo']+'-'+inputs['s_dist']+'_d'+inputs['dynamics']
    dir3 = '_'+inputs['loss_fn']+'_c'+str(inputs['critic_mean_type'])+'_'+str(buff_exp-1)+'b'+str(int(inputs['buffer']))[0:2]
    dir4 = '_m'+str(inputs['multi_steps']) +'_'+str(step_exp-1)+'s'+str(int(inputs['n_cumsteps']))[0:2]+'_n'+str(round+1)

    directory = dir1 + dir2 + dir3 + dir4

    return directory

def cauchy(estimated: T.FloatTensor, target: T.FloatTensor, scale: float) -> T.FloatTensor:
    """
    Cauchy loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        scale: Cauchy scale parameter

    Returns:
        loss (float): loss values
    """
    scale = T.tensor(scale)    
    arg = ((target-estimated)/scale)**2

    return T.log(1 + arg)

def nagy_algo(estimated: T.FloatTensor, target: T.FloatTensor, scale: float) -> float:
    """
    Use the Nagy alogrithm to estimate the Cauchy scale paramter based on residual errors based on
    Eq. 18 in http://www.jucs.org/jucs_12_9/parameter_estimation_of_the/jucs_12_09_1332_1344_nagy.pdf
    and Section 3.2 in https://tongliang-liu.github.io/papers/TPAMITruncatedNMF.pdf.
    
    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        scale (float>0): current Cauchy scale parameter [step: t]
        
    Returns:
        scale_new: updated scale parameter > 0 [step: t + 1]
    """
    scale = T.tensor(scale)
    estimated, target = estimated.detach().clone(), target.detach().clone()
    arg = ((target-estimated)/scale)**2
    arg2 = 1/(1 + arg)
    error = T.mean(arg2)
    inv_error = 1/error
    
    if inv_error >= 1:
        return (scale * T.sqrt(inv_error - 1)).detach().cpu().numpy()
    else:
        return scale.detach().cpu().numpy()

def correntropy(estimated: T.FloatTensor, target: T.FloatTensor, kernel: float) -> T.FloatTensor:
    """
    Correntropy-Induced Metric (CIM) loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        kernel (float>0): width of Gaussain

    Returns:
        loss (float): loss values
    """   
    kernel = T.tensor(kernel)
    arg = (target-estimated)**2

    return (1 - T.exp(-arg/(2 * kernel**2)) / T.sqrt(2 * np.pi * kernel))

def cim_size(estimated: T.FloatTensor, target: T.FloatTensor) -> float:
    """
    Empirically estimated kernel size for CIM taken as the average reconstruction error
    based on Eq. 25 in  https://lcs.ios.ac.cn/~ydshen/ICDM-12.pdf.

    Parameters:
        estimated (list): current Q-values
        target (list): Q-values from mini-batch

    Returns:
        kernel (float>0): standard deviation
    """
    arg = (target-estimated)**2
    kernel = T.std(arg.detach().clone(), unbiased=False)

    return kernel.cpu().numpy()

def hypersurface(estimated: T.FloatTensor, target: T.FloatTensor) -> T.FloatTensor:
    """
    Hypersurface cost based loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        loss (float): loss values
    """    
    arg = (target-estimated)**2

    return (T.sqrt(1 + arg) - 1) 

def mse(estimated: T.FloatTensor, target: T.FloatTensor, exp:int =0) -> T.FloatTensor:
    """
    MSE loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        exp (even int): exponent in addition to MSE

    Returns:
        loss (float): loss values
    """
    return (target-estimated)**(int(2 + exp))
    
def mae(estimated: T.FloatTensor, target: T.FloatTensor) -> T.FloatTensor:
    """
    MAE loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        loss (float): loss values
    """
    return T.abs(target-estimated)

def huber(estimated: T.FloatTensor, target: T.FloatTensor) -> T.FloatTensor:
    """
    Huber loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        loss (float): loss values
    """
    arg = (target - estimated)
    loss = T.where(T.abs(arg) < 1, 0.5 * arg**2, T.abs(arg) - 0.5)

    return loss

def hill_est(values: T.FloatTensor) -> T.FloatTensor:
    """
    Calculates using extreme value thoery the Hill estimator as a proxy for the tail index 
    of a power law provided alpha > 0. Treats all values as extreme.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation

    Returns:
        alpha: tail index of power law
    """
    values = T.abs(values.view(-1))

    order_stats = values.sort(descending=True)[0]
    min_val = T.log(order_stats[0])
    vals = T.log(order_stats[1:])

    hill_1 = (vals - min_val).mean()
    gamma = hill_1 
        
    # method of moments
    # hill_2 = ((vals - min_val)**2).mean()
    # gamma += 1 - 1 / 2 * (1 - hill_1**2 / hill_2)**(-1)

    return 1 / gamma

def zipf_plot(values: T.FloatTensor, zipf_x: T.FloatTensor, zipf_x2: T.FloatTensor) -> T.FloatTensor:
    """
    Obtain gradient of Zipf plot created with order statistics.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis

    Returns:
        alpha: tail index of power law
    """
    values = values.view(-1)

    order_stats = values.sort(descending=True)[0]
    order_stats = T.log(order_stats)
    diff_stats = order_stats - T.mean(order_stats)
    
    gamma = T.sum(zipf_x * diff_stats) / zipf_x2

    return 1 / gamma

def aggregator(values: T.FloatTensor, shadow_low_mul: T.FloatTensor, shadow_high_mul: T.FloatTensor, 
               zipf_x: T.FloatTensor, zipf_x2: T.FloatTensor) -> Tuple[T.FloatTensor, T.FloatTensor, 
               T.FloatTensor, T.FloatTensor]:
    """
    Aggregates mini-batch values with the `empirical' mean (SLLN approach) and using  
    heuristics constructs a power law with an estimated tail exponent to infer a shadow mean.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        shadow_low_mul: lower bound multiplier of minimum for which values greater than are of interest
        shadow_high_mul: finite improbable upper bound multiplier of maximum of distributions
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis
    
    Returns:
        mean: empirical mean
        min: minimum critic loss
        max: maximum critic loss
        shadow: shadow mean
        alpha: tail index of power law
    """
    mean = T.mean(values)
    min = T.min(values)
    max = T.max(values)
    low = T.min(values) * shadow_low_mul
    high = T.max(values) * shadow_high_mul

    # alpha = hill_est(values)
    alpha = zipf_plot(values, zipf_x, zipf_x2)

    # upper incomplete gamma function valid only for alpha, high > 0
    up_gamma = T.exp(T.lgamma(1 - alpha)) * (1 - T.igamma(1 - alpha, alpha / high))

    # shadow mean point estimate
    shadow = low + (high - low) * T.exp(alpha / high) * (alpha / high)**alpha * up_gamma

    return mean, min, max, shadow, alpha

def loss_function(estimated: T.FloatTensor, target: T.FloatTensor, shadow_low_mul: T.FloatTensor, 
                  shadow_high_mul: T.FloatTensor, zipf_x: T.FloatTensor, zipf_x2: T.FloatTensor, 
                  loss_type: str, scale: float, kernel: float) -> Tuple[T.FloatTensor, T.FloatTensor, 
                  T.FloatTensor, T.FloatTensor]:
    """
    Gives scalar critic loss value retaining graph for backpropagation.
    
    Parameters:
        estimated: current Q-values from mini-batch
        target: raget Q-values from mini-batch
        shadow_low_mul: lower bound multiplier of minimum for which values greater than are of interest
        shadow_high_mul: finite improbable upper bound multiplier of maximum of distributions
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis
        loss_type: loss function title
        scale (float>0): current Cauchy scale parameter
        kernel (float>0): standard deviation for CIM 

    Returns:
        mean: empirical mean
        min: minimum critic loss
        max: maximum critic loss
        shadow: shadow mean
        alpha: tail index of power law
    """
    if loss_type == "MSE":
        values = mse(estimated, target, 0)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "Huber":
        values = huber(estimated, target)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "MAE":
        values = mae(estimated, target)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "HSC":
        values = hypersurface(estimated, target)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "Cauchy":
        values = cauchy(estimated, target, scale)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "CIM":
        values = correntropy(estimated, target, kernel)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type[0:3] == "MSE" and type(int(loss_type[3:])) == int:
        values = mse(estimated, target, int(loss_type[3:]))
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

def shadow_means(alpha: np.ndarray, min: np.ndarray, max: np.ndarray, min_mul: float, 
                 max_mul: float) -> np.ndarray:
    """
    Construct shadow mean given tail exponent and sample min/max for various multipliers

    Parameters:
        alpha: tail indices
        min: sample minimum critic loss
        max: sample maximum critic loss
        min_mul: min multiplier
        max_mul: max multiplier

    Returns:
        shadow: shadow mean
    """
    low = min * min_mul
    high = max * max_mul

    up_gamma = T.exp(T.lgamma(1 - alpha)) * (1 - T.igamma(1 - alpha, alpha / high))
    shadow = low + (high - low) * T.exp(alpha / high) * (alpha / high)**alpha * up_gamma

    return shadow