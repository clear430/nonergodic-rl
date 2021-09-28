import numpy as np
import scipy.optimize as op
import scipy.special as sp
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
        
    if '_n2_' in env:
        dim += 2
    if '_n10_' in env:
        dim += 10

    if 'Market_' in env:
        dim += 10
        
    if '_SH_' in env:
        dim = 4 + 2 + 1

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
    sh = 1 if safe_haven == True else 0

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

def mul_inv_n_summary(mul_inputs: dict, aggregate_n: np.ndarray, safe_haven: bool =False) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        keqv: max multiplier for equvilance between shadow and empirical means
        lev_sh: leverage for safe haven
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
    keqv = np.zeros((ninv, count_x, count_z * 2)) 

    reward = np.zeros((ninv, count_x, count_y))
    lev = np.zeros((ninv, count_x, count_y))
    stop = np.zeros((ninv, count_x, count_y))
    reten = np.zeros((ninv, count_x, count_y))
    lev_sh = np.zeros((ninv, count_x, count_y))

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

                        if safe_haven == True:
                            lev_sh[i, t, s + n * int(mul_inputs['n_eval'])] = aggregate_n[i, n, t, s, 26]

    shadow[np.isnan(shadow)] = loss[np.isnan(shadow)]

    for i in range(ninv):
        for t in range(count_x):
            for n in range(mul_inputs['n_trials'] * 2):
                keqv[i, t, n] = shadow_equiv(loss[i, t, n], tail[i, t, n], 
                                             lmin[i, t, n], loss[i, t, n], 1)

    return reward, lev, stop, reten, loss, tail, shadow, cmax, keqv, lev_sh