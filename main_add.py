from algos.algo_sac import Agent_sac
from algos.algo_td3 import Agent_td3
from datetime import datetime
import extras.plots as plots
import extras.utils as utils
import gym
import numpy as np
import os
import pybullet_envs
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

inputs = {
    # SAC hyperparameters
    'sac_actor_learn_rate': 3e-4,               # actor learning rate (Adam optimiser)
    'sac_critic_learn_rate': 3e-4,              # critic learning rate (Adam optimiser)
    'sac_temp_learn_rate': 3e-4,                # log temperature learning rate (Adam optimiser)
    'sac_layer_1_units': 256,                   # nodes in first fully connected layer
    'sac_layer_2_units': 256,                   # nodes in second fully connected layer
    'sac_actor_step_update': 1,                 # actor policy network update frequency (steps)
    'sac_temp_step_update': 1,                  # temperature update frequency (steps)
    'sac_target_critic_update': 1,              # target critic networks update frequency (steps)
    'initial_logtemp': 0,                       # log weighting given to entropy maximisation
    'reward_scale': 1,                          # constant scaling factor of next reward ('inverse temperature')
    'reparam_noise': 1e-6,                      # miniscule constant to keep logarithm bounded

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
    'n_trials': 3,                              # number of total unique training trials
    'n_cumsteps': 3e3,                          # maximum cumulative steps per trial (must be greater than warmup)
    'eval_freq': 1e3,                           # interval of steps between evaluation episodes
    'max_eval_reward': 1e4,                     # maximum reward per evaluation episode
    'n_eval': 1e2                               # number of evalution episodes
    }

gym_envs = {
        # ENV_KEY: [env_id, input_dim, action_dim, intial warmup steps (generate random seed)]

        # OpenAI Box2D continuous control tasks
        '0': ['LunarLanderContinuous-v2', 8, 2, 1e3], 
        '1': ['BipedalWalker-v3', 24, 4, 1e3],              
        '2': ['BipedalWalkerHardcore-v3', 24, 4, 1e3],
        # Roboschool environments ported to PyBullet
        '3': ['CartPoleContinuousBulletEnv-v0', 4, 1, 1e3], 
        '4': ['InvertedPendulumBulletEnv-v0', 5, 1, 1e3],
        '5': ['InvertedDoublePendulumBulletEnv-v0', 9, 1, 1e3], 
        '6': ['HopperBulletEnv-v0', 15, 3, 1e3], 
        '7': ['Walker2DBulletEnv-v0', 22, 6, 1e3],
        '8': ['HalfCheetahBulletEnv-v0', 26, 6, 1e4],    # composed of a single training episode (can't use multi-steps)
        '9': ['AntBulletEnv-v0', 28, 8, 1e4],            # composed of a single training episode (can't use multi-steps)
        '10': ['HumanoidBulletEnv-v0', 44, 17, 1e4], 
        # KOD*LAB quadruped direct-drive legged robots ported to PyBullet
        '11': ['MinitaurBulletEnv-v0', 28, 8, 1e4],
        # DeepMimic simulation of a imitating Humanoid mimic ported to PyBullet
        '12': ['HumanoidDeepMimicWalkBulletEnv-v1', 197, 36, 1e4],
        '13': ['HumanoidDeepMimicBackflipBulletEnv-v1', 197, 36, 1e4]
        }

ENV_KEY = 7
algo_name = ['TD3', 'SAC']                # off-policy models 'SAC', 'TD3'
surrogate_critic_loss = ['MSE']    # 'MSE', 'Huber', 'MAE', 'HSC', 'Cauchy', 'CIM', 'MSE2', 'MSE4', 'MSE6'
multi_steps = [1]                  # 1, 3, 5, 7 (any positive integer > 0)

env = gym.make(gym_envs[str(ENV_KEY)][0])
inputs = {'input_dims': env.observation_space.shape, 'num_actions': env.action_space.shape[0], 
          'max_action': env.action_space.high[0], 'min_action': env.action_space.low[0], 
          'env_id': gym_envs[str(ENV_KEY)][0], 'random': gym_envs[str(ENV_KEY)][3], 
          'loss_fn': 'MSE', 'algo': 'TD3', **inputs}
env = env.env    # allow access to setting enviroment state and remove episode step limit

if __name__ == '__main__':

    for algo in algo_name:
        for loss_fn in surrogate_critic_loss:
            for mstep in multi_steps:

                inputs['loss_fn'], inputs['algo'], inputs['multi_steps'] = loss_fn.upper(), algo.upper(), mstep
                trial_log = np.zeros((inputs['n_trials'], int(inputs['n_cumsteps']), 19))
                eval_log = np.zeros((inputs['n_trials'], int(inputs['n_cumsteps'] / inputs['eval_freq']), int(inputs['n_eval']), 20))

                for round in range(inputs['n_trials']):

                    directory = utils.save_directory(inputs, round)
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
                        done, step = False, 0
                        score = 0 if inputs['dynamics'] == 'A' else inputs['initial_reward']

                        while not done:
                            action, _ = agent.select_next_action(state)
                            next_state, reward, done, info = env.step(action)
                            agent.store_transistion(state, action, reward, next_state, done)

                            # gradient update interval (perform backpropagation)
                            if cum_steps % int(inputs['grad_step'][inputs['algo']]) == 0:
                                loss, logtemp, loss_params = agent.learn()

                            state = next_state
                            score += reward
                            step += 1
                            cum_steps += 1
                            end_time = time.perf_counter()

                            # conduct periodic agent evaluation episodes without learning
                            if cum_steps % int(inputs['eval_freq']) == 0:
                                utils.eval_policy(agent, inputs, eval_log, cum_steps, round, eval_run, loss, logtemp, loss_params)
                                eval_run += 1

                            if cum_steps > int(inputs['n_cumsteps']-1):
                                break

                        time_log.append(end_time - start_time)
                        score_log.append(score)
                        step_log.append(step)
                        loss_log.append(loss)
                        logtemp_log.append(logtemp)
                        loss_params_log.append(loss_params)

                        # save actor-critic neural network weights for checkpointing
                        trail_score = np.mean(score_log[-inputs['trail']:])
                        if trail_score > best_score:
                            best_score = trail_score
                            agent.save_models()
                            print('New high trailing score!')

                        print('{} {}-{}-{}-{} ep/st/cst {}/{}/{} {:1.0f}/s: r {:1.0f}, tr{} {:1.0f}, C/Cm/Cs {:1.1f}/{:1.1f}/{:1.0f}, a/c/k {:1.2f}/{:1.2f}/{:1.2f}, A/T {:1.1f}/{:1.2f}'
                        .format(datetime.now().strftime('%d %H:%M:%S'), 
                                inputs['algo'], inputs['s_dist'], inputs['loss_fn'], round+1,  episode, step, cum_steps, step/time_log[-1], 
                                score, inputs['trail'], trail_score,  np.mean(loss[0:2]), np.mean(loss[4:6]), np.mean(loss[6:8]),
                                np.mean(loss[8:10]), np.mean(loss_params[0:2]), np.mean(loss_params[2:4]),  loss[8], np.exp(logtemp)))
                        # rl_algorithm-sampling_sitribution-loss_function-trial,  ep/st/cst = episode/steps/cumulative_steps,  /s = training_steps_per_second,
                        # r = episode_reward, tr = trailing_episode_reward,  C/Cm/Cs = avg_critic_loss/max_critic_loss/shadow_critic_loss
                        # c/k/a = avg_Cauchy_scale/avg_CIM_kernel_size/avg_tail_exponent,  A/T = avg_actor_loss/sac_entropy_temperature

                        episode += 1

                    count = len(score_log)
                    trial_log[round, :count, 0], trial_log[round, :count, 1] =  time_log, score_log
                    trial_log[round, :count, 2], trial_log[round, :count, 3:14] = step_log, loss_log
                    trial_log[round, :count, 14], trial_log[round, :count, 15:] = logtemp_log, loss_params_log

                    if not os.path.exists('./results/'+inputs['env_id']):
                        os.makedirs('./results/'+inputs['env_id'])

                    if inputs['n_trials'] == 1:
                        plots.plot_learning_curve(inputs, trial_log[round], directory+'.png')

                # truncate training trial log array up to maximum episodes
                count_episodes = [np.min(np.where(trial_log[trial, :, 0] == 0)) for trial in range(int(inputs['n_trials']))]
                max_episode = np.max(count_episodes) 
                trial_log = trial_log[:, :max_episode, :]

                np.save(directory+'_trial.npy', trial_log)
                np.save(directory+'_eval.npy', eval_log)

                if inputs['n_trials'] > 1:
                    plots.plot_trial_curve(inputs, trial_log, directory+'_trial.png')    # plot of agent training with linear interpolation across all trials
                    # plots.plot_eval_curve(inputs, eval_log, directory+'_eval.png')       # plot of agent evaluation round scores across all trials
                    plots.plot_eval_loss_2d(inputs, eval_log, directory+'_2d.png')       # plot of agent evaluation round scores and training critic losses across all trials
                    plots.plot_eval_loss_3d(inputs, eval_log, directory+'_3d.png')       # 3D plot of agent evaluation round scores and training critic losses across all trials