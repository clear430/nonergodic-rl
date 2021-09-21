import numpy as np
import scipy.optimize as op
import scipy.special as sp
import torch as T
from typing import Tuple

def save_directory(inputs: dict, results: bool) -> str:
    """
    Provides string directory for data and plot saving names.

    Parameters:
        inputs: dictionary containg all execution details
        results: whether results (True) or model (False)

    Returns:
        directory: file path and name to give to current experiment plots
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    dir = ['./results/', 
           'additive/' if inputs['dynamics'] == 'A' else 'multiplicative/',
           inputs['env_id']+'/',
           inputs['env_id']+'--',
           inputs['dynamics']+'_',
           inputs['algo']+'-',
           inputs['s_dist'],
           '_'+inputs['loss_fn'],
           '-'+str(inputs['critic_mean_type']),
           '_B'+str(int(inputs['buffer']))[0:2]+'e'+str(buff_exp-1),
           '_M'+str(inputs['multi_steps']),
           '_S'+str(int(inputs['n_cumsteps']))[0:2]+'e'+str(step_exp-1),
           '_N'+str(inputs['n_trials'])  
           ]

    if results == False:
        dir[0] = './models/'

    directory = ''.join(dir)

    return directory

def plot_subtitles(inputs: dict):
    """
    Generate subtitles for plots and figures.

    Parameters:
        inputs: dictionary containg all execution details
    
    Returns:
        sub: subtitle to be used in plots
    """
    step_exp = int(len(str(int(inputs['n_cumsteps']))) - 1)
    buff_exp = int(len(str(int(inputs['buffer']))) - 1)

    sub = [inputs['env_id']+'--',
           inputs['dynamics']+'_',
           inputs['algo']+'-',
           inputs['s_dist'],
           '_'+inputs['loss_fn'],
           '-'+str(inputs['critic_mean_type']),
           '_B'+str(int(inputs['buffer']))[0:2]+'e'+str(buff_exp-1),
           '_M'+str(inputs['multi_steps']),
           '_S'+str(int(inputs['n_cumsteps']))[0:2]+'e'+str(step_exp-1),
           '_N'+str(inputs['n_trials'])  
           ]
    
    sub = ''.join(sub)
    
    return sub

def multi_log_dim(inputs: dict) -> int:
    """
    Generates risk-related parameter log dimension for multiplicative experiments 
    with dimensions dependent on the environment characteristics.

    Parameters
        inputs: dictionary containg all execution details
        
    Returns:
        dim: dimensions for log array
    """
    env = inputs['env_id']
    
    dim = 4
    
    if '_InvB' in env:
        dim += 1
    if '_InvC' in env:
        dim += 2
    
    if '_SH_' in env:
        dim += 3

    if '_n2_' in env:
        dim += 2
    if '_n10_' in env:
        dim += 10
    
    return dim

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

def truncation(estimated: T.FloatTensor, target: T.FloatTensor) \
        -> Tuple[T.FloatTensor, T.FloatTensor]:
    """
    Elements to be truncated based on Gaussian distribution assumption based on a 
    correction of Section 3.3 in https://arxiv.org/pdf/1906.00495.pdf.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        estimated: truncated current Q-values
        target: truncated Q-values from mini-batch
    """  
    sigma1, mean1 = T.std_mean(estimated, unbiased=False)
    sigma2, mean2 = T.std_mean(target, unbiased=False)
    zero1 = estimated - estimated
    zero2 = target - target
    
    # 3-sigma rule
    estimated = T.where(T.abs(estimated - mean1) > 3 * sigma1, zero1, estimated)   
    target = T.where(T.abs(target - mean2) > 3 * sigma2, zero2, target)

    return estimated, target

def cauchy(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor, scale: float) \
        -> T.cuda.FloatTensor:
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

def nagy_algo(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor, scale: float) \
        -> float:
    """
    Use the Nagy alogrithm to estimate the Cauchy scale paramter based on residual errors based on
    Eq. 18 in http://www.jucs.org/jucs_12_9/parameter_estimation_of_the/jucs_12_09_1332_1344_nagy.pdf
    and Section 3.2 in https://arxiv.org/pdf/1906.00495.pdf.
    
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

def correntropy(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor, kernel: float) \
        -> T.cuda.FloatTensor:
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

def cim_size(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor) -> float:
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

def hypersurface(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor) \
        -> T.cuda.FloatTensor:
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

def mse(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor, exp:int =0) \
        -> T.cuda.FloatTensor:
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
    
def mae(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor) \
        -> T.cuda.FloatTensor:
    """
    MAE loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        loss (float): loss values
    """
    return T.abs(target-estimated)

def huber(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor) \
        -> T.cuda.FloatTensor:
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

def hill_est(values: T.cuda.FloatTensor) -> T.cuda.FloatTensor:
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
    min_val = order_stats[0]
    geo_mean = T.prod(order_stats[1:])
    geo_mean = geo_mean**(1 / geo_mean.shape[0])

    hill_1 = T.log(geo_mean / min_val)
    gamma = hill_1 
        
    # method of moments
    # hill_2 = ((vals - min_val)**2).mean()
    # gamma += 1 - 1 / 2 * (1 - hill_1**2 / hill_2)**(-1)

    return 1 / gamma

def zipf_plot(values: T.cuda.FloatTensor, zipf_x: T.cuda.FloatTensor, zipf_x2: T.cuda.FloatTensor) \
        -> T.cuda.FloatTensor:
    """
    Obtain gradient of Zipf (or Pareto Q-Q) plot using ordered statistics.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis

    Returns:
        alpha (>=0): tail index estimated using plot gradient
    """
    values = values.view(-1)

    order_stats = values.sort(descending=True)[0]
    order_stats = T.log(order_stats)
    diff_stats = order_stats - T.mean(order_stats)
    
    # standard linear regression coefficient
    gamma = T.sum(zipf_x * diff_stats) / zipf_x2

    return 1 / gamma

def aggregator(values: T.cuda.FloatTensor, shadow_low_mul: T.cuda.FloatTensor, 
               shadow_high_mul: T.cuda.FloatTensor, zipf_x: T.cuda.FloatTensor, 
               zipf_x2: T.cuda.FloatTensor) \
        -> Tuple[T.cuda.FloatTensor, T.cuda.FloatTensor, T.cuda.FloatTensor, 
                 T.cuda.FloatTensor, T.cuda.FloatTensor]:
    """
    Aggregates several mini-batch summary statistics: 'empirical' mean (strong LLN approach), min/max, 
    uses power law heuristics to estimate the shadow mean, and the tail exponent.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        shadow_low_mul: lower bound multiplier of sample minimum to form minimum threshold of interest
        shadow_high_mul: upper bound multiplier of sample maximum for tail distributions
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis
    
    Returns:
        mean: empirical mean
        min: minimum critic loss
        max: maximum critic loss
        shadow: shadow mean
        alpha: tail index of power law
    """
    mean, min, max = T.mean(values), T.min(values), T.max(values)
    
    low, high = T.min(values) * shadow_low_mul, T.max(values) * shadow_high_mul
    alpha = zipf_plot(values, zipf_x, zipf_x2)

    # upper incomplete gamma function valid only for alpha, high > 0
    up_gamma = T.exp(T.lgamma(1 - alpha)) * (1 - T.igamma(1 - alpha, alpha / high))

    # shadow mean estimate
    shadow = low + (high - low) * T.exp(alpha / high) * (alpha / high)**alpha * up_gamma

    return mean, min, max, shadow, alpha

def loss_function(estimated: T.cuda.FloatTensor, target: T.cuda.FloatTensor, 
                  shadow_low_mul: T.cuda.FloatTensor, shadow_high_mul: T.cuda.FloatTensor, 
                  zipf_x: T.cuda.FloatTensor, zipf_x2: T.cuda.FloatTensor, 
                  loss_type: str, scale: float, kernel: float) \
        -> Tuple[T.cuda.FloatTensor, T.cuda.FloatTensor, T.cuda.FloatTensor, 
                 T.cuda.FloatTensor, T.cuda.FloatTensor]:
    """
    Gives scalar critic loss value (retaining graph) for network backpropagation.
    
    Parameters:
        estimated: current Q-values from mini-batch
        target: raget Q-values from mini-batch
        shadow_low_mul: lower bound multiplier of sample minimum to form minimum threshold of interest
        shadow_high_mul: upper bound multiplier of maximum of distributions
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
        alpha: tail index
    """
    if loss_type == "MSE":
        values = mse(estimated, target, 0)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul,
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "HUB":
        values = huber(estimated, target)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, 
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "MAE":
        values = mae(estimated, target)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, 
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "HSC":
        values = hypersurface(estimated, target)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, 
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "CAU":
        values = cauchy(estimated, target, scale)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, 
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "TCA":
        estimated, target = truncation(estimated, target)
        values = cauchy(estimated, target, scale)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, 
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type == "CIM":
        values = correntropy(estimated, target, kernel)
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, 
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

    elif loss_type[0:3] == "MSE" and type(int(loss_type[3:])) == int:
        values = mse(estimated, target, int(loss_type[3:]))
        mean, min, max, shadow, alpha = aggregator(values, shadow_low_mul, shadow_high_mul, 
                                                   zipf_x, zipf_x2)
        return mean, min, max, shadow, alpha

def shadow_means(alpha: np.ndarray, min: np.ndarray, max: np.ndarray, 
                 min_mul: float, max_mul: float) \
        -> np.ndarray:
    """
    Construct shadow mean given the tail exponent and sample min/max for varying multipliers.

    Parameters:
        alpha: sample tail index
        min: sample minimum critic loss
        max: sample maximum critic loss
        low_mul: lower bound multiplier of sample minimum to form minimum threshold of interest
        max_mul: upper bound multiplier of maximum of distributions

    Returns:
        shadow: shadow mean
    """
    low, high = min * min_mul, max * max_mul
    up_gamma = sp.gamma(1 - alpha) * sp.gammaincc(1 - alpha, alpha / high)
    shadow = low + (high - low) * np.exp(alpha / high) * (alpha / high)**alpha * up_gamma

    return shadow

def shadow_equiv(mean: np.ndarray, alpha: np.ndarray, min: np.ndarray, 
                 max: np.ndarray, min_mul: float =1) \
        -> np.ndarray:
    """
    Estimate max multiplier required for equivalence between empirical (arthmetic) mean
    and shadow mean estimate.

    Parameters:
        mean: empirical mean
        alpha: sample tail index
        min: sample minimum critic loss
        max: sample maximum critic loss
        low_mul: lower bound multiplier of sample minimum to form minimum threshold of interest

    Returns:
        max_mul:upper bound multiplier of maximum of distributions for equivalent
    """
    # select intial guess of equivilance multiplier
    x0 = 1

    if alpha < 1:
        f = lambda max_mul: shadow_means(alpha, min, max, min_mul, max_mul) - mean
        max_mul_solve = op.root(f, x0, method='hybr')
        max_mul_solve = max_mul_solve.x
    else:
        max_mul_solve = x0

    return max_mul_solve

def mul_inv_aggregate(env_keys: list, gym_envs: dict, mul_inputs: dict, safe_haven: bool =False) \
        -> np.ndarray:
    """
    Combine environment evaluation data for investors across the same number of assets.

    Parameters:
        env_keys: list of environments
        gym_envvs: dictionary of all environment details
        mul_inputs: dictionary of execution parameters
        safe_have: whether investor is using insurance safe haven

    Retuens:
        eval: aggregated evaluation data across all investors
    """
    sh = 3 if safe_haven == True else 0

    mul_nsteps = str(int(mul_inputs['n_cumsteps']))[0:2]
    mul_nstep_exp = str(int(len(str(int(mul_inputs['n_cumsteps']))) - 1) - 1)

    for key in env_keys:
        name = [gym_envs[str(key)][0] for key in env_keys]
        path = ['./results/multiplicative/' + n + '/' for n in name]

        eval = np.zeros((len(name), int(mul_inputs['n_trials']), 
                            int(mul_inputs['n_cumsteps'] / mul_inputs['eval_freq']), 
                            int(mul_inputs['n_eval']), 20 + 16 + sh))
        num = 0
        for env in name:
            data_path = path[num]+env+'--M_TD3-N_MSE-E_B10e5_M1_S'+mul_nsteps+'e'+mul_nstep_exp+ \
                        '_N'+str(int(mul_inputs['n_trials']))

            file1 = np.load(data_path+'_eval.npy')
            file2 = np.load(data_path+'_eval_risk.npy')
            file = np.concatenate((file1, file2), axis=3)

            eval[num, :, :, :, :20 + file2.shape[3]] = file

            num += 1

    return eval

def mul_inv_n_summary(mul_inputs: dict, aggregate_n: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seperate aggregate array into variables.
    
    Parameters:
        mul_inputs: dictionary of execution parameters
        aggregate_n: aggregated evaluation data across all investors

    Retuens:
        reward: 1 + time-average growth rate
        lev: leverages
        stop: stop-losses
        reten: retention ratios
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        heqv: max multiplier for equvilance between shadow and empirical means
    """
    ninv = aggregate_n.shape[0]

    count_x = int(mul_inputs['n_cumsteps'] / mul_inputs['eval_freq'])
    count_y = int(mul_inputs['n_trials'] * int(mul_inputs['n_eval']))
    count_z = int(mul_inputs['n_trials'] )

    loss = np.zeros((ninv, count_x, count_z * 2))
    shadow = np.zeros((ninv, count_x, count_z * 2))
    tail = np.zeros((ninv, count_x, count_z * 2))
    lmin = np.zeros((ninv, count_x, count_z * 2))
    cmax = np.zeros((ninv, count_x, count_z * 2))
    heqv = np.zeros((ninv, count_x, count_z * 2)) 

    reward = np.zeros((ninv, count_x, count_y))
    lev = np.zeros((ninv, count_x, count_y))
    stop = np.zeros((ninv, count_x, count_y))
    reten = np.zeros((ninv, count_x, count_y))

    for i in range(ninv):
        for t in range(count_x):
            for n in range(mul_inputs['n_trials']):
                
                    loss[i, t, (n * 2):(n * 2) + 2 ] = aggregate_n[i, n, t, 0, 3:5]
                    shadow[i, t, (n * 2):(n * 2) + 2 ] = aggregate_n[i, n, t, 0, 9:11]
                    tail[i, t, (n * 2):(n * 2) + 2 ] = aggregate_n[i, n, t, 0, 11:13]
                    lmin[i, t, (n * 2):(n * 2) + 2 ] = aggregate_n[i, n, t, 0, 5:7]
                    cmax[i, t, (n * 2):(n * 2) + 2 ] = aggregate_n[i, n, t, 0, 7:9]

                    for s in range(int(mul_inputs['n_eval'])):
                        reward[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 20]
                        lev[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 23]
                        stop[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 24]
                        reten[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 25]

    shadow[np.isnan(shadow)] = loss[np.isnan(shadow)]

    for i in range(ninv):
        for t in range(count_x):
            for n in range(mul_inputs['n_trials'] * 2):
                heqv[i, t, n] = shadow_equiv(loss[i, t, n], tail[i, t, n], 
                                             lmin[i, t, n], loss[i, t, n], 1)

    return reward, lev, stop, reten, loss, tail, shadow, cmax, heqv
