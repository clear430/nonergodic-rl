from scripts.exp_additive import additive_env
from scripts.exp_multiplicative import multiplicative_env

# off-policy models: 'SAC', 'TD3'
algo_name = ['TD3']
# critic loss function: 'MSE', 'Huber', 'MAE', 'HSC', 'Cauchy', 'CIM', 'MSE2', 'MSE4', 'MSE6'
critic_loss = ['MSE']
# bootstrapping: additive = integer > 0, multiplicative = 1 
multi_steps = [1]

ENV_KEY = 14

gym_envs = {
    # ENV_KEY: [env_id, input_dim, action_dim, intial warmup steps to generate random seed]

    # ADDITIVE ENVIRONMENTS
    # - M=1: composed of a single training episode (can't use multi-steps)
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
    '8': ['HalfCheetahBulletEnv-v0', 26, 6, 1e4],    # (M=1)
    '9': ['AntBulletEnv-v0', 28, 8, 1e4],            # (M=1)
    '10': ['HumanoidBulletEnv-v0', 44, 17, 1e4], 
    # KOD*LAB quadruped direct-drive legged robots ported to PyBullet
    '11': ['MinitaurBulletEnv-v0', 28, 8, 1e4],
    # DeepMimic simulation of a imitating Humanoid mimic ported to PyBullet
    '12': ['HumanoidDeepMimicWalkBulletEnv-v1', 197, 36, 1e4],
    '13': ['HumanoidDeepMimicBackflipBulletEnv-v1', 197, 36, 1e4],

    # MULTIPLICATVE ENVIRONMENTS
    # assets following the equally likely +50%/-40% gamble
    # investor 1: portfolio of one, two and ten assets
    '14': ['Investor1_1x', 2, 1, 3e3],
    '15': ['Investor1_2x', 3, 2, 1e3],
    '16': ['Investor1_10x', 11, 10, 1e3],
    # investor 2: portfolio of one, two and ten assets
    '17': ['Investor2_1x', 2, 2, 1e3],
    '18': ['Investor2_2x', 3, 3, 1e3],
    '19': ['Investor2_10x', 11, 11, 1e3],
    # investor 3: portfolio of one, two and ten assets
    '20': ['Investor3_1x', 2, 3, 1e3],
    '21': ['Investor3_2x', 3, 4, 1e3],
    '22': ['Investor3_10x', 11, 12, 1e3],

    # assets following GBM
    # investor 1: portfolio of one, two and ten assets 
    '23': ['Investor1GBM_1x', 2, 1, 1e3],
    '24': ['Investor1GBM_2x', 3, 2, 1e3],
    '25': ['Investor1GBM_10x', 11, 10, 1e3]
    }

inputs_dict = {
    # execution parameters
    'n_trials': 3,                              # number of total unique training trials
    'n_cumsteps': 4e4,                          # maximum cumulative steps per trial (must be greater than warmup)
    'eval_freq': 1e3,                           # interval of steps between evaluation episodes
    'max_eval_reward': 1e4,                     # maximum reward per evaluation episode
    'n_eval': 1e2,                              # number of evalution episodes

    # learning variables
    'buffer': 1e6,                              # maximum transistions in experience replay buffer
    'discount': 0.99,                           # discount factor for successive steps            
    'multi_steps': 1,                           # bootstrapping of target critic values and discounted rewards
    'trail': 50,                                # moving average of training episode scores used for model saving
    'cauchy_scale': 1,                          # Cauchy scale parameter initialisation value
    'r_abs_zero': None,                         # defined absolute zero value for rewards
    'continue': False,                          # whether to continue learning with same parameters across trials

    # critic loss aggregation
    'critic_mean_type': 'E',                    # critic mean estimation method either empirical 'E' or shadow 'S' 
    'shadow_low_mul': 0e0,                      # lower bound multiplier of mini critic difference power law  
    'shadow_high_mul': 1e1,                     # upper bound multiplier of max critic difference power law

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

    'algo_name': algo_name,
    'critic_loss': critic_loss,
    'multi_steps': multi_steps
    }

if __name__ == '__main__':

    if ENV_KEY <= 13:
        additive_env(gym_envs=gym_envs, inputs=inputs_dict, ENV_KEY=ENV_KEY)
    else:
        multiplicative_env(gym_envs=gym_envs, inputs=inputs_dict, ENV_KEY=ENV_KEY)