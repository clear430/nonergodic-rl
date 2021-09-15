import sys
sys.path.append("./")

import numpy as np
import os
import extras.plots_multiverse as plots_multiverse
import time
import torch as T
from torch.distributions.bernoulli import Bernoulli
from typing import Tuple

VRAM = 1  # do you have >= 8GB of VRAM?
if VRAM == 1:
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
else:
    device = T.device('cpu' if T.cuda.is_available() else 'cpu')
    # still need 8GB of free RAM or reduce number of investors

dir = 'results/multiverse/'    # directory for saving numpy arrays

INVESTORS = 1.5e5           # number of random investors
HORIZON = 3e3               # total time steps
TOP= INVESTORS * 1e-4       # define top performers
VALUE_0 = 1e2               # intial portfolio value of each investor
UP_PROB = 0.5               # probability of up move
UP_R = 0.5                  # upside return (>=0)
DOWN_R = -0.4               # downside return (0<=)
ASYM_LIM = 1e-12            # offset to enforce 'optimal' leverage bound

INVESTORS = T.tensor(int(INVESTORS), dtype=T.int32, device=device)
HORIZON = T.tensor(int(HORIZON), dtype=T.int32, device=device)
VALUE_0 = T.tensor(VALUE_0, device=device)
TOP = int(TOP) if TOP > 1 else int(1)    # minimum 1 person in the top sample
ASYM_LIM = T.tensor(ASYM_LIM, device=device)

# theoretical optimal leverage based on 'expectations'
BIGGER_PAYOFF = np.abs(DOWN_R) if np.abs(UP_R) > np.abs(DOWN_R) else -np.abs(UP_R) 
LEV_FACTOR = T.tensor(1 / BIGGER_PAYOFF, device=device)
LEV_FACTOR = LEV_FACTOR - ASYM_LIM if np.abs(UP_R) > np.abs(DOWN_R) else LEV_FACTOR + ASYM_LIM

def param_range(low: float, high: float, increment: float):
    """
    Create list of increments.

    Parameters:
        lows: start
        high: end
        increment: step size
    """
    min = int(low/increment)
    max = int(high/increment + 1 + 1e-4)    # minor offset to counter Python strangness

    return [x * increment for x in range(min, max, 1)]

def fixed_final_lev(outcomes: T.FloatTensor, top: int, value_0: T.FloatTensor, up_r: T.FloatTensor, 
                    down_r: T.FloatTensor, lev_low: float, lev_high: float, lev_incr: float):
    """
    Simple printing of end result of fixed leverage valuations.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes
    """
    lev_range = np.array(param_range(lev_low, lev_high, lev_incr))
    lev_range = - lev_range if -down_r > up_r else lev_range

    for lev in lev_range:
        
        gambles = T.where(outcomes == 1, 1 + lev * up_r, 1 + lev * down_r)

        value_T = value_0 * gambles.prod(dim = 1)

        sort_value = value_T.sort(descending=True)[0]
        top_value = sort_value[0:top]
        adj_value = sort_value[top:]

        # summary statistics
        std, mean = T.std_mean(value_T, unbiased=False)
        med = T.median(value_T)
        mad = T.mean(T.abs(value_T - mean))

        std_top, mean_top = T.std_mean(top_value, unbiased=False)
        med_top = T.median(top_value)
        mad_top = T.mean(T.abs(top_value - mean_top))
        
        std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
        med_adj = T.median(adj_value)
        mad_adj = T.mean(T.abs(adj_value - mean_adj))

        print("""       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}"""
                 .format(lev*100, mean, med, mad, std, mean_top, med_top, mad_top, std_top, 
                         mean_adj, med_adj, mad_adj, std_adj))

def smart_lev(outcomes: T.FloatTensor, investors: T.IntTensor, horizon: T.IntTensor, top: int, 
              value_0: T.FloatTensor, up_r: T.FloatTensor, down_r: T.FloatTensor, lev_low: float, 
              lev_high: float, lev_incr: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Valuations across all time for fixed leverages.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes
    
    Returns:
        data: valuation summary statistics for each time step
        data_T: valuations at  maturity
    """
    lev_range = np.array(param_range(lev_low, lev_high, lev_incr))
    lev_range = - lev_range if -down_r > up_r else lev_range


    data = T.zeros((len(lev_range), 3 * 4 + 1, horizon - 1))
    data_T = T.zeros((len(lev_range), investors))

    i = 0
    for lev in lev_range:

        gambles = T.where(outcomes == 1, 1 + lev * up_r, 1 + lev * down_r)
        initial = value_0 * gambles[:, 0]

        for t in range(horizon - 1):

            value_t = initial * gambles[:, t + 1]
            initial = value_t

            sort_value = value_t.sort(descending=True)[0]
            top_value = sort_value[0:top]
            adj_value = sort_value[top:]

            # summary statistics
            std, mean = T.std_mean(value_t, unbiased=False)
            med = T.median(value_t)
            mad = T.mean(T.abs(value_t - mean))

            std_top, mean_top = T.std_mean(top_value, unbiased=False)
            med_top = T.median(top_value)
            mad_top = T.mean(T.abs(top_value - mean_top))
            
            std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
            med_adj = T.median(adj_value)
            mad_adj = T.mean(T.abs(adj_value - mean_adj))

            data[i, :, t] = T.tensor([mean, mean_top, mean_adj, mad, mad_top, mad_adj, 
                                      std, std_top, std_adj, med, med_top, med_adj, lev])

        data_T[i, :] = value_t 

        i += 1

        print("""       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}"""
                 .format(lev*100, mean, med, mad, std, mean_top, med_top, mad_top, std_top, 
                         mean_adj, med_top, mad_adj, std_adj))

    return data.cpu().numpy(), data_T.cpu().numpy()

def optimal_lev(value_t: float, value_0: float, value_min: float, lev_factor: float, roll: float) -> float:
    """
    Calculate optimal leverage besed on either rolling or fixed stop loss at each time step.

    Parameters:
        value_t: current portfolio values
        value_0: intitial portfolio value
        value_min: global stop-loss
        lev_factor: maximum leverage to not be stopped out by a single move
        roll: retention ratio
    """
    if roll == 0:
        rolling_loss = value_min
        value_roll = T.maximum(value_min, rolling_loss)
        opt_lev = lev_factor * (1 - value_roll / value_t)

    else:
        rolling_loss = T.where(value_t <= value_0, value_min, value_0 + roll * (value_t - value_0)) 
        opt_lev = lev_factor * (1 - rolling_loss / value_t)
        
    return opt_lev

def big_brain_lev(outcomes: T.FloatTensor, investors: T.IntTensor, horizon: T.IntTensor, 
                  top: int, value_0: T.FloatTensor, up_r: T.FloatTensor, down_r: T.FloatTensor, 
                  lev_factor: float, stop_min: float, stop_max: float, stop_incr: float, 
                  roll_max: float, roll_min: float, roll_incr: float) \
        -> np.ndarray:
    """
    Valuations across all time for variable stop-losses and retention ratios that calculates optimal leverage
    at each time step for each investor.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        lev_factor: maximum leverage to not be stopped out by a single move
        stop_low: starting stop-loss
        stop_high: ending stop-loss
        stop_incr: stop-loss step sizes
        roll_low: starting retention ratio
        roll_high: ending retention ratio
        roll_incr: retention ratio step sizes
    
    Returns:
        data: valuation summary statistics for each time step
    """
    stop_range = param_range(stop_min, stop_max, stop_incr)
    roll_range = param_range(roll_min, roll_max, roll_incr)

    data = T.zeros((len(roll_range), len(stop_range), 3 * 4 * 2 + 2, horizon - 1))

    j = 0
    for roll_level in roll_range:

        i = 0
        for stop_level in stop_range:
            
            gambles = T.where(outcomes == 1, up_r, down_r)
            
            value_min = stop_level * value_0
            
            lev = optimal_lev(value_0, value_0, value_min, lev_factor, roll_level)
            sample_lev = T.ones((1, investors), device=device) * lev
            
            initial = value_0 * (1 + lev * gambles[:, 0])

            sample_lev = optimal_lev(initial, value_0, value_min, lev_factor, roll_level)

            for t in range(horizon - 1):
                sort_lev = sample_lev.sort(descending=True)[0]
                top_lev = sort_lev[0:top]
                adj_lev = sort_lev[top:]

                # leverage summary statistics
                lstd, lmean = T.std_mean(sample_lev, unbiased=False)
                lmed = T.median(sample_lev)
                lmad = T.mean(T.abs(sample_lev - lmean))

                lstd_top, lmean_top = T.std_mean(top_lev, unbiased=False)
                lmed_top = T.median(top_lev)
                lmad_top = T.mean(T.abs(top_lev - lmean_top))
                
                lstd_adj, lmean_adj = T.std_mean(adj_lev, unbiased=False)
                lmed_adj = T.median(adj_lev)
                lmad_adj = T.mean(T.abs(adj_lev - lmean_adj))

                data[j, i, 12:26, t] = T.tensor([lmean, lmean_top, lmean_adj, lmad, lmad_top, lmad_adj, 
                                                 lstd, lstd_top, lstd_adj, lmed, lmed_top, lmed_adj, 
                                                 stop_level, roll_level])

                # calculate one-period change in valuations
                value_t = initial * (1 + sample_lev * gambles[:, t + 1])
                initial = value_t

                sample_lev = optimal_lev(initial, value_0, value_min, lev_factor, roll_level)
                
                sort_value = value_t.sort(descending=True)[0]
                top_value = sort_value[0:top]
                adj_value = sort_value[top:]

                # valuation summary statistics
                std, mean = T.std_mean(value_t, unbiased=False)
                med = T.median(value_t)
                mad = T.mean(T.abs(value_t - mean))

                std_top, mean_top = T.std_mean(top_value, unbiased=False)
                med_top = T.median(top_value)
                mad_top = T.mean(T.abs(top_value - mean_top))
                
                std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
                med_adj = T.median(adj_value)
                mad_adj = T.mean(T.abs(adj_value - mean_adj))

                data[j, i, 0:12, t] = T.tensor([mean, mean_top, mean_adj, mad, mad_top, mad_adj, 
                                                std, std_top, std_adj, med, med_top, med_adj])

            i += 1

            print("""stop/roll {:1.0f}/{:1.0f}%: 
                    avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}
                    top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}
                    adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}"""
                    .format(stop_level*100, roll_level*100, mean, med, mad, std, lmean, lmed, lmad, lstd, 
                            mean_top, med_top, mad_top, std_top, lmean_top, lmed_top, lmad_top, lstd_top, 
                            mean_adj, med_adj, mad_adj, std_adj, lmean_adj, lmed_adj, lmad_adj, lstd_adj))
        j += 1

    return data.cpu().numpy()

def galaxy_brain_lev(ru_min: float, ru_max: float, ru_incr: float, rd_min: float, rd_max: float, 
                     rd_incr: float, pu_min: float, pu_max: float, pu_incr: float) \
        -> np.ndarray:
    """
    Optimal leverage determined using the Kelly criterion to maximise the geometric return of a binary payout
    valid across all time steps.

    Parameters:
        ru_min: starting up return
        ru_max: ending up return
        ru_incr: up return step sizes
        rd_min: starting down return (absolute)
        rd_max: ending down return (absolute)
        rd_incr: down return step sizes (absolute)
        pu_min: starting up probability
        pu_max: ending up probability
        pu_incr: up probability step sizes
    """
    ru_range = param_range(ru_min, ru_max, ru_incr)
    rd_range = param_range(rd_min, rd_max, rd_incr)
    pu_range = param_range(pu_min, pu_max, pu_incr)

    data = np.zeros((len(pu_range), len(ru_range), len(ru_range), 3 + 1))

    i= 0
    for pu in pu_range:

        j = 0
        for ru in ru_range:

            k = 0
            for rd in rd_range:
                kelly = pu / rd - (1 - pu) / ru
                data[i, j, k, :] = [pu, ru, rd, kelly]

                k += 1

            j += 1

        i += 1

    return data

if __name__ == '__main__': 

    ## run experiments

    # start_time = time.perf_counter()

    # T.manual_seed(420)    # set fixed seed for reproducibility
    # probabilites = Bernoulli(UP_PROB)
    # outcomes = probabilites.sample(sample_shape=(INVESTORS, HORIZON)).to(device)

    # fixed_final_lev(outcomes, TOP, VALUE_0, UP_R, DOWN_R, lev_low=0.05, lev_high=1.0, lev_incr=0.05)

    # inv1_val_data, inv1_val_data_T = smart_lev(outcomes, INVESTORS, HORIZON, TOP, VALUE_0, UP_R, DOWN_R,
    #                                            lev_low=0.1, lev_high=1, lev_incr=0.1)
    
    # inv2_val_data = big_brain_lev(outcomes, INVESTORS, HORIZON, TOP, VALUE_0, UP_R, DOWN_R, LEV_FACTOR 
    #                               stop_min=0.1, stop_max=0.1, stop_incr=0.1, roll_max=0.0,
    #                               roll_min=0.0, roll_incr=0.1)

    # inv3_val_data = big_brain_lev(outcomes, INVESTORS, HORIZON, TOP, VALUE_0, UP_R, DOWN_R, LEV_FACTOR, 
    #                               stop_min=0.05, stop_max=0.95, stop_incr=0.05, roll_max=0.95,
    #                               roll_min=0.70, roll_incr=0.05)

    # inv4_lev_data = galaxy_brain_lev(ru_min=0.2, ru_max=0.8, ru_incr=0.005, rd_min=0.2, rd_max=0.8, 
    #                                  rd_incr=0.005, pu_min=0.25, pu_max=0.75, pu_incr=0.25)
    
    # end_time = time.perf_counter()
    # print('time: {:1.1f}'.format(end_time-start_time))

    ## save experiment data

    # if not os.path.exists('./' + dir):
    #     os.makedirs('./' + dir)

    # np.save(dir + 'inv1_val.npy', inv1_val_data)
    # np.save(dir + 'inv1_val_T.npy', inv1_val_data_T)
    # np.save(dir + 'inv2_val.npy', inv2_val_data)
    # np.save(dir + 'inv3_val.npy', inv3_val_data)
    # np.save(dir + 'inv4_lev.npy', inv4_lev_data)

    # load experiment data and save figures

    if not os.path.exists('./docs/figs'):
            os.makedirs('./docs/figs')

    inv4_lev_data = np.load(dir + '/inv4_lev.npy')
    plots_multiverse.plot_inv4(inv4_lev_data, 'docs/figs/inv4.png')

    inv3_val_data = np.load(dir + 'inv3_val.npy')
    plots_multiverse.plot_inv3(inv3_val_data, 'docs/figs/inv3.png')

    inv2_val_data = np.load(dir + 'inv2_val.npy')
    plots_multiverse.plot_inv2(inv2_val_data, 'docs/figs/inv2.png')

    inv1_val_data = np.load(dir + 'inv1_val.npy')
    inv1_val_data_T = np.load(dir + 'inv1_val_T.npy')
    plots_multiverse.plot_inv1(inv1_val_data, inv1_val_data_T, 'docs/figs/inv1.png')