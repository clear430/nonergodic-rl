import sys
sys.path.append("./")

from algos.algo_sac import Agent_sac
from algos.algo_td3 import Agent_td3
from datetime import datetime
import envs.coin_flip_envs as coin_flip_envs
# import envs.dice_roll_envs as dice_roll_envs
# import envs.gbm_envs as gbm_envs
# import extras.plots_multiplicative as plots
import extras.utils as utils
import numpy as np
import os
import time

assert hasattr(Agent_sac, 'select_next_action'), 'missing agent action selection'
assert hasattr(Agent_sac, 'store_transistion'), 'missing transition storage functionality'
assert hasattr(Agent_sac, 'learn'), 'missing agent learning functionality'
assert hasattr(Agent_sac, 'save_models'), 'missing agent save functionality'
assert hasattr(Agent_sac, 'load_models'), 'missing agent load functionality'
assert hasattr(Agent_td3, 'select_next_action'), 'missing agent action selection'
assert hasattr(Agent_td3, 'store_transistion'), 'missing transition storage functionality'
assert hasattr(Agent_td3, 'learn'), 'missing agent learning functionality'
assert hasattr(Agent_td3, 'save_models'), 'missing agent save functionality'
assert hasattr(Agent_td3, 'load_models'), 'missing agent load functionality'

def multiplicative_env(gym_envs: dict, inputs: dict, ENV_KEY: int):
    """
    Conduct experiments for multiplicative environments.
    """
    if ENV_KEY <= 22:
        env = eval('coin_flip_envs.'+gym_envs[str(ENV_KEY)][0]+'()')
    else:
        env = eval('dice_roll_envs.'+gym_envs[str(ENV_KEY)][0]+'()')

    inputs = {'input_dims': env.observation_space.shape, 'num_actions': env.action_space.shape[0], 
            'max_action': env.action_space.high.min(), 'min_action': env.action_space.low.max(),    # assume all elements span equal domain 
            'env_id': gym_envs[str(ENV_KEY)][0], 'random': gym_envs[str(ENV_KEY)][3], 
            'dynamics': 'M',    # gambling dynamics 'M' (multiplicative)
            'loss_fn': 'MSE', 'algo': 'TD3', **inputs}

    for algo in inputs['algo_name']:
        for loss_fn in inputs['critic_loss']:
            for mstep in inputs['multi_steps']:

                inputs['loss_fn'], inputs['algo'], inputs['multi_steps'] = loss_fn.upper(), algo.upper(), mstep

                for round in range(inputs['n_trials']):

                    time_log, score_log, step_log, logtemp_log, loss_log, loss_params_log = [], [], [], [], [], []
                    cum_steps, eval_run, episode = 0, 0, 1
                    best_score = env.reward_range[0]
                    if inputs['continue'] == True:
                        inputs['initial_logtemp'] = logtemp if round > 1 else False    # load existing SAC parameter to continue learning

                    agent = Agent_td3(env, inputs) if inputs['algo'] == 'TD3' else Agent_sac(env, inputs)
                    if inputs['continue'] == True:
                        agent.load_models() if round > 1 else False    # load existing actor-critic parameters to continue learning

                    while cum_steps < int(inputs['n_cumsteps']):
                        start_time = time.perf_counter()            
                        state = env.reset()
                        done, step, score = False, 0, 0

                        while not done:
                            action, _ = agent.select_next_action(state)
                            next_state, reward, done, risk = env.step(action)
                            agent.store_transistion(state, action, reward, next_state, done)

                            if cum_steps % int(inputs['grad_step'][inputs['algo']]) == 0:
                                loss, logtemp, loss_params = agent.learn()

                            state = next_state
                            score += reward
                            step += 1
                            cum_steps += 1
                            end_time = time.perf_counter()

                            time_log.append(end_time - start_time)
                            score_log.append(score)
                            step_log.append(step)
                            loss_log.append(loss)
                            logtemp_log.append(logtemp)
                            loss_params_log.append(loss_params)

                            print('ep/st/cst {}/{}/{} {:1.0f}/s: V/g/[risk] ${:1.6f}/{:1.6f}%/{}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}'
                                  .format(episode, step, cum_steps, step/time_log[-1], risk[0], risk[1]-1, np.round(risk[2:]*100, 0), np.mean(loss[0:2]), np.mean(loss[4:6]), 
                                          np.mean(loss[6:8]), np.mean(loss[8:10]), np.mean(loss_params[0:2]), np.mean(loss_params[2:4]), loss[8]+3, np.exp(logtemp)+5))

                            if cum_steps > int(inputs['n_cumsteps']-1):
                                break

                        episode += 1