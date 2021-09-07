from algo_td3 import Agent_td3
from datetime import datetime
import gym
import numpy as np
import os
import plots
import pybullet_envs
import multi_envs
import time
import utils

assert hasattr(Agent_td3, 'select_next_action'), 'missing agent action selection'
assert hasattr(Agent_td3, 'store_transistion'), 'missing transition storage functionality'
assert hasattr(Agent_td3, 'learn'), 'missing agent learning functionality'
assert hasattr(Agent_td3, 'save_models'), 'missing agent save functionality'
assert hasattr(Agent_td3, 'load_models'), 'missing agent load functionality'

inputs = {
    # TD3 hyperparameters          
    'td3_actor_learn_rate': 1e-3,               # ibid.
    'td3_critic_learn_rate': 1e-3,              # ibid.
    'td3_layer_1_units': 400,                   # ibid.
    'td3_layer_2_units': 300,                   # ibid.
    'td3_actor_step_update': 2,                 # ibid.
    'td3_target_actor_update': 2,               # target actor network update frequency (steps)
    'td3_target_critic_update': 2,              # ibid.
    'policy_noise': 0.1,                        # Gaussian exploration noise added to next actions
    'target_policy_noise': 0.2,                 # Gaussian noise added to next target actions as a regulariser
    'target_policy_clip': 0.5,                  # Clipping of Gaussian noise added to next target actions

    # shared parameters
    'target_update_rate': 5e-3,                 # Polyak averaging rate for target network parameter updates
    's_dist': 'N',                              # actor policy sampling via 'L' (Laplace) or 'N' (Normal) distribution 
    'batch_size': {'SAC': 256, 'TD3': 100},     # mini-batch size
    'grad_step': {'SAC': 1, 'TD3': 1},          # standard gradient update frequency (steps)

    # learning variables
    'buffer': 1e6,                              # maximum transistions in experience replay buffer
    'discount': 0.99,                           # discount factor for successive steps            
    'multi_steps': 1,                           # bootstrapping of target critic values and discounted rewards
    'trail': 50,                                # moving average of training episode scores used for model saving
    'cauchy_scale': 1,                          # Cauchy scale parameter initialisation value
    'continue': False,                          # whether to continue learning with same parameters across trials

    # critic loss aggregation
    'critic_mean_type': 'E',                    # critic mean estimation method either empirical 'E' or shadow 'S' 
    'shadow_low_mul': 0e0,                      # lower bound multiplier of minimum for critic power law  
    'shadow_high_mul': 1e1,                     # finite improbable upper bound multiplier of maximum for critic difference power law

    # ergodicity
    'dynamics': 'A',                            # gambling dynamics either 'A' (additive) or 'M' (multiplicative)
    'game_over': 0.99,                          # threshold for ending episode for all cumualtive rewards
    'initial_reward': 1e1,                      # intial cumulative reward value of each episode
    'unique_hist': 'Y',                         # whether each step in episode creates 'Y' or 'N' a unique history
    'compounding': 'N',                         # if multiplicative, whether compounding 'Y' or 'N' multi-steps 
    'r_abs_zero': None,                         # defined absolute zero value for rewards
     
    # execution parameters
    'n_trials': 1,                              # number of total unique training trials
    'n_cumsteps': 4e4,                          # maximum cumulative steps per trial (must be greater than warmup)
    'eval_freq': 9e5,                           # interval of steps between evaluation episodes
    'max_eval_reward': 1e4,                     # maximum reward per evaluation episode
    'n_eval': 1e2                               # number of evalution episodes
    }

gym_envs = {
        # ENV_KEY: [env_id, input_dim, action_dim, intial warmup steps (generate random seed)]

        # single asset investor categories for the equally likely +50%/-40% gamble
        '1': ['SimpleInv1_1x', 1, 1, 3e3], 
        '2': ['SimpleInv2_1x', 1, 2, 3e3],           
        '3': ['SimpleInv3_1x', 1, 3, 3e3],
        
        # double asset investor categories for the equally likely +50%/-40% gamble
        '4': ['SimpleInv1_2x', 3, 2, 3e3],
        '5': ['SimpleInv2_2x', 3, 3, 3e3],
        '6': ['SimpleInv3_2x', 3, 4, 3e3]
        }

ENV_KEY = 6
algo_name = ['TD3']                # off-policy model 'TD3'
surrogate_critic_loss = ['MSE']    # 'MSE', 'Huber', 'MAE', 'HSC', 'Cauchy', 'CIM', 'MSE2', 'MSE4', 'MSE6'
multi_steps = [1]                  # 1

if ENV_KEY == 1:
    env = multi_envs.Investor1_1x()
elif ENV_KEY == 2:
    env = multi_envs.Investor2_1x()
elif ENV_KEY == 3:
    env = multi_envs.Investor3_1x()
elif ENV_KEY == 4:
    env = multi_envs.Investor1_2x()
elif ENV_KEY == 5:
    env = multi_envs.Investor2_2x()
elif ENV_KEY == 6:
    env = multi_envs.Investor3_2x()

inputs = {'input_dims': env.observation_space.shape, 'num_actions': env.action_space.shape[0], 
          'max_action': env.action_space.high.min(), 'min_action': env.action_space.low.max(),    # assume all elements span equal domain 
          'env_id': gym_envs[str(ENV_KEY)][0], 'random': gym_envs[str(ENV_KEY)][3], 
          'loss_fn': 'MSE', 'algo': 'TD3', **inputs}

if __name__ == '__main__':

    for algo in algo_name:
        for loss_fn in surrogate_critic_loss:
            for mstep in multi_steps:

                inputs['loss_fn'], inputs['algo'], inputs['multi_steps'] = loss_fn.upper(), algo.upper(), mstep

                for round in range(inputs['n_trials']):

                    time_log, score_log, step_log, logtemp_log, loss_log, loss_params_log = [], [], [], [], [], []
                    cum_steps, eval_run, episode = 0, 0, 1
                    best_score = env.reward_range[0]

                    agent = Agent_td3(env, inputs)

                    while cum_steps < int(inputs['n_cumsteps']):
                        start_time = time.perf_counter()            
                        state = env.reset()
                        done, step = False, 0
                        score = 0 if inputs['dynamics'] == 'A' else inputs['initial_reward']

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

                            print('ep/st/cst {}/{}/{} {:1.0f}/s: V/g/[risk] ${}/{:1.6f}%/{}, C/Cm/Cs {:1.1f}/{:1.1f}/{:1.0f}, a/c/k/A {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}'
                                  .format(episode, step, cum_steps, step/time_log[-1], state[0], reward*100, np.round(risk*100, 0), np.mean(loss[0:2]), 
                                          np.mean(loss[4:6]), np.mean(loss[6:8]),np.mean(loss[8:10]), np.mean(loss_params[0:2]), np.mean(loss_params[2:4]), loss[8]))

                            if cum_steps > int(inputs['n_cumsteps']-1):
                                break

                        episode += 1