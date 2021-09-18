import numpy as np
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
           '_S'+str(int(inputs['n_cumsteps']))[0]+'e'+str(step_exp),
           '_N'+str(inputs['n_trials'])  
           ]

    if results == False:
        dir[0] = 'models/'

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
           '_S'+str(int(inputs['n_cumsteps']))[0]+'e'+str(step_exp),
           '_N'+str(inputs['n_trials'])  
           ]
    
    sub = ''.join(sub)
    
    return sub

def multi_trial_log(inputs: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates risk-rlatd parameter log for multiplicative experiments with
    dimensions dependent on the environment.

    Parameters
        inputs: dictionary containg all execution details
        
    Returns:
        trial_risk_log: array of zeros for agent learning logs
        eval_risk_log:  array of zeros for agent evaluation logs
    """

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

def truncation(estimated: T.FloatTensor, target: T.FloatTensor) -> Tuple[T.FloatTensor, T.FloatTensor]:
    """
    Elements to be truncated based on Gaussian distribution assumption based on a correction of
    Section 3.3 in https://arxiv.org/pdf/1906.00495.pdf.

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
    min_val = T.log(order_stats[0])
    vals = T.log(order_stats[1:])

    hill_1 = (vals - min_val).mean()
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
        min_mul: min multiplier to form shadow mean low threshold
        max_mul: max multiplier to form shadow mean high estimate

    Returns:
        shadow: shadow mean
    """
    low, high = min * min_mul, max * max_mul
    up_gamma = sp.gamma(1 - alpha) * sp.gammaincc(1 - alpha, alpha / high)
    shadow = low + (high - low) * np.exp(alpha / high) * (alpha / high)**alpha * up_gamma

    return shadow