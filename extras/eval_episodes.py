from datetime import datetime
import gym
import numpy as np
import pybullet_envs
import time
from typing import Tuple

def additive(agent: object, inputs: dict, eval_log: np.ndarray, cum_steps: int, round: int, 
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
    inputs['algo'], inputs['s_dist'], inputs['loss_fn'], round+1, int(inputs['n_eval']), cum_steps))

    print('{} Training Summary: T/Cg/Cs {:1.2f}/{:1.2f}/{:1.2f}, C/A {:1.1f}/{:1.1f}'
    .format(datetime.now().strftime('%d %H:%M:%S'), np.exp(logtemp), sum(loss_params[0:2])/2, 
        sum(loss_params[2:4])/2, sum(loss[0:2]), loss[-1]))
    
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