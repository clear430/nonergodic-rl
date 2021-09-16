from algos.algo_sac import Agent_sac
from algos.networks_sac import ActorNetwork as ActorNetwork_sac
from algos.networks_sac import CriticNetwork as CriticNetwork_sac
from algos.algo_td3 import Agent_td3
from algos.networks_td3 import ActorNetwork as ActorNetwork_td3
from algos.networks_td3 import CriticNetwork as CriticNetwork_td3
from extras.replay import ReplayBuffer
from scripts.exp_additive import additive_env
from scripts.exp_multiplicative import multiplicative_env

# off-policy models: ['SAC', 'TD3']
algo_name = ['TD3']
# critic loss functions: ['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'CIM', 'MSE2', 'MSE4', 'MSE6']
critic_loss = ['MSE']
# bootstrapping of target critic values and discounted rewards: [list of integers > 0] 
multi_steps = [1]

ENV_KEY = 14

gym_envs = {
    # ENV_KEY: [env_id, state_dim, action_dim, intial warm-up steps to generate random seed]

    #### ADDITIVE ENVIRONMENTS

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
    '8': ['HalfCheetahBulletEnv-v0', 26, 6, 1e4],
    '9': ['AntBulletEnv-v0', 28, 8, 1e4],
    '10': ['HumanoidBulletEnv-v0', 44, 17, 1e4], 
    # KOD*LAB quadruped direct-drive legged robots ported to PyBullet
    '11': ['MinitaurBulletEnv-v0', 28, 8, 1e4],
    # DeepMimic simulation of a imitating Humanoid mimic ported to PyBullet
    '12': ['HumanoidDeepMimicWalkBulletEnv-v1', 197, 36, 1e4],
    '13': ['HumanoidDeepMimicBackflipBulletEnv-v1', 197, 36, 1e4],

    #### MULTIPLICATVE ENVIRONMENTS

    # assets following the equally likely +50%/-40% gamble
    '14': ['Coin_n1_Inv1', 2, 1, 1e3], '15': ['Coin_n2_Inv1', 3, 2, 1e3], '16': ['Coin_n10_Inv1', 11, 10, 1e3],
    '17': ['Coin_n1_Inv2', 2, 2, 1e3], '18': ['Coin_n2_Inv2', 3, 3, 1e3], '19': ['Coin_n10_Inv2', 11, 11, 1e3],
    '20': ['Coin_n1_Inv3', 2, 3, 1e3], '21': ['Coin_n2_Inv3', 3, 4, 1e3], '22': ['Coin_n10_Inv3', 11, 12, 1e3],
    # assets following the dice roll
    '23': ['Dice_n1_Inv1', 2, 1, 1e3], '24': ['Dice_n2_Inv1', 3, 2, 1e3], '25': ['Dice_n10_Inv1', 11, 10, 1e3],
    '26': ['Dice_n1_Inv2', 2, 2, 1e3], '27': ['Dice_n2_Inv2', 3, 3, 1e3], '28': ['Dice_n10_Inv2', 11, 11, 1e3],
    '29': ['Dice_n1_Inv3', 2, 3, 1e3], '30': ['Dice_n2_Inv3', 3, 4, 1e3], '31': ['Dice_n10_Inv3', 11, 12, 1e3],
    # assets following dice roll with insurance safe haven
    '32': ['Dice_SH_n1_Inv1', 2, 1, 1e3], '33': ['Dice_SH_n2_Inv1', 3, 2, 1e3], '34': ['Dice_SH_n10_Inv1', 11, 10, 1e3],
    '35': ['Dice_SH_n1_Inv2', 2, 2, 1e3], '36': ['Dice_SH_n2_Inv2', 3, 3, 1e3], '37': ['Dice_SH_n10_Inv2', 11, 11, 1e3],
    '38': ['Dice_SH_n1_Inv3', 2, 3, 1e3], '39': ['Dice_SH_n2_Inv3', 3, 4, 1e3], '40': ['Dice_SH_n10_Inv3', 11, 12, 1e3],
    # assets following GBM
    '41': ['GBM_n1_Inv1', 2, 1, 1e3], '42': ['GBM_n2_Inv1', 3, 2, 1e3], '43': ['GBM_n10_Inv1', 11, 10, 1e3],
    '44': ['GBM_n1_Inv2', 2, 2, 1e3], '45': ['GBM_n2_Inv2', 3, 3, 1e3], '46': ['GBM_n10_Inv2', 11, 11, 1e3],
    '47': ['GBM_n1_Inv3', 2, 3, 1e3], '48': ['GBM_n2_Inv3', 3, 4, 1e3], '49': ['GBM_n10_Inv3', 11, 12, 1e3],
    # assets following GBM with insurance safe haven
    '50': ['GBM_SH_n1_Inv1', 2, 1, 1e3], '51': ['GBM_SH_n2_Inv1', 3, 2, 1e3], '52': ['GBM_SH_n10_Inv1', 11, 10, 1e3],
    '53': ['GBM_SH_n1_Inv2', 2, 2, 1e3], '54': ['GBM_SH_n2_Inv2', 3, 3, 1e3], '55': ['GBM_SH_n10_Inv2', 11, 11, 1e3],
    '56': ['GBM_SH_n1_Inv3', 2, 3, 1e3], '57': ['GBM_SH_n2_Inv3', 3, 4, 1e3], '58': ['GBM_SH_n10_Inv3', 11, 12, 1e3],
    }

inputs_dict = {
    # execution parameters
    'n_trials': 2,                              # number of total unique training trials
    'n_cumsteps': 2e3,                          # maximum cumulative steps per trial (must be greater than warm-up)
    'eval_freq': 1e3,                           # interval of steps between evaluation episodes
    'n_eval': 1e1,                              # number of evalution episodes
    'max_eval_reward': 1e4,                     # maximum reward per evaluation episode
    'max_eval_steps': 1e0,                      # maximum steps per evaluation episode for multiplicative environments

    # learning variables
    'buffer': 1e6,                              # maximum transistions in experience replay buffer
    'discount': 0.99,                           # discount factor for successive steps
    'trail': 50,                                # moving average of training episode scores used for model saving
    'cauchy_scale': 1,                          # Cauchy scale parameter initialisation value
    'actor_percentile': 1,                      # bottom percentile of actor mini-batch to be maximised (>0, <=1)
    'r_abs_zero': None,                         # defined absolute zero value for rewards
    'continue': False,                          # whether to continue learning with same parameters across trials

    # critic loss aggregation
    'critic_mean_type': 'E',                    # critic network learning either empirical 'E' or shadow 'S' (only E) 
    'shadow_low_mul': 1e0,                      # lower bound multiplier of critic difference power law  
    'shadow_high_mul': 1e1,                     # upper bound multiplier of critic difference power law

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

    # environment details
    'algo_name': [algo.upper() for algo in algo_name],
    'critic_loss': [loss.upper() for loss in critic_loss],
    'multi_steps': multi_steps,
    'ENV_KEY': ENV_KEY
    }

# CONDUCT TESTS
gte0 = 'must be greater than or equal to 0'
gt0 = 'must be greater than 0'
gte1 = 'must be greater than or equal to 1'

# execution tests
assert isinstance(inputs_dict['n_trials'], (float, int)) and \
    int(inputs_dict['n_trials']) >= 1, gte1
assert isinstance(inputs_dict['n_cumsteps'], (float, int)) and \
    int(inputs_dict['n_cumsteps']) >= 1, gte1
assert isinstance(inputs_dict['eval_freq'], (float, int)) and \
    int(inputs_dict['eval_freq']) >= 1, gte1
assert isinstance(inputs_dict['n_eval'], (float, int)) and \
    int(inputs_dict['n_eval']) >= 1, gte1
assert isinstance(inputs_dict['max_eval_reward'], (float, int)) and \
    inputs_dict['max_eval_reward'] > 0, gt0
assert isinstance(inputs_dict['max_eval_steps'], (float, int)) and \
    int(inputs_dict['max_eval_steps']) >= 1, gte1

# learning varaible tests
assert isinstance(inputs_dict['buffer'], (float, int)) and \
    int(inputs_dict['buffer']) >= 1 and \
        inputs_dict['buffer'] >= inputs_dict['n_cumsteps'], \
            'must be greater than or equal to both 1 and n_cumsteps'
assert inputs_dict['discount'] >= 0 \
    and inputs_dict['discount'] < 1, 'should be within [0, 1)'
assert isinstance(inputs_dict['trail'], (float, int)) and \
    int(inputs_dict['trail']) >= 1, gte1
assert isinstance(inputs_dict['cauchy_scale'], (float, int)) and \
    inputs_dict['cauchy_scale'] > 0, gt0
assert isinstance(inputs_dict['actor_percentile'], (float, int)) and \
    inputs_dict['actor_percentile'] > 0 and \
        inputs_dict['actor_percentile'] <= 1, 'must be within (0, 1]'
assert isinstance(inputs_dict['r_abs_zero'], (float, int)) or inputs_dict['r_abs_zero'] == None, \
    'either real number or None'
assert isinstance(inputs_dict['continue'], bool), 'must be either True or False'

# critic loss aggregation tests
assert inputs_dict['critic_mean_type'] == 'E' or 'S', 'must be either "E" or "S"'
assert isinstance(inputs_dict['shadow_low_mul'], (float, int)) and \
    inputs_dict['shadow_low_mul'] >= 0, gte0
assert isinstance(inputs_dict['shadow_high_mul'], (float, int)) and \
    inputs_dict['shadow_high_mul'] > 0, gt0

# SAC hyperparameter tests
assert isinstance(inputs_dict['sac_actor_learn_rate'], (float, int)) and \
    inputs_dict['sac_actor_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['sac_critic_learn_rate'], (float, int)) and \
    inputs_dict['sac_critic_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['sac_temp_learn_rate'], (float, int)) and \
    inputs_dict['sac_temp_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['sac_layer_1_units'], (float, int)) and \
    int(inputs_dict['sac_layer_1_units']) >= 1, gte1
assert isinstance(inputs_dict['sac_layer_2_units'], (float, int)) and \
    int(inputs_dict['sac_layer_2_units']) >= 1, gte1
assert isinstance(inputs_dict['sac_actor_step_update'], (float, int)) and \
    int(inputs_dict['sac_actor_step_update']) >= 1, gte1
assert isinstance(inputs_dict['sac_temp_step_update'], (float, int)) and \
    int(inputs_dict['sac_temp_step_update']) >= 1, gte1
assert isinstance(inputs_dict['sac_target_critic_update'], (float, int)) and \
    int(inputs_dict['sac_target_critic_update']) >= 1, gte1
assert isinstance(inputs_dict['initial_logtemp'], (float, int)), 'must be any real number'
assert isinstance(inputs_dict['reparam_noise'], float) and \
    inputs_dict['reparam_noise'] > 1e-7 and \
        inputs_dict['reparam_noise'] < 1e-5, 'must be any real and near the vicinity of 1e-6'

# TD3 hyperparameter tests
assert isinstance(inputs_dict['td3_actor_learn_rate'], (float, int)) and \
    inputs_dict['td3_actor_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['td3_critic_learn_rate'], (float, int)) and \
    inputs_dict['td3_critic_learn_rate'] > 0, gt0
assert isinstance(inputs_dict['td3_layer_1_units'], (float, int)) and \
    int(inputs_dict['td3_layer_1_units']) >= 1, gte1
assert isinstance(inputs_dict['td3_layer_2_units'], (float, int)) and \
    int(inputs_dict['td3_layer_2_units']) >= 1, gte1
assert isinstance(inputs_dict['td3_actor_step_update'], (float, int)) and \
    int(inputs_dict['td3_actor_step_update']) >= 1, gte1
assert isinstance(inputs_dict['td3_target_actor_update'], (float, int)) and \
    int(inputs_dict['td3_target_actor_update']) >= 1, gte1
assert isinstance(inputs_dict['td3_target_critic_update'], (float, int)) and \
    int(inputs_dict['td3_target_critic_update']) >= 1, gte1
assert isinstance(inputs_dict['td3_target_critic_update'], (float, int)) and \
    int(inputs_dict['td3_target_critic_update']) >= 1, gte1
assert isinstance(inputs_dict['policy_noise'], (float, int)) and \
    inputs_dict['policy_noise'] >= 0, gte0
assert isinstance(inputs_dict['target_policy_noise'], (float, int)) and \
    inputs_dict['target_policy_noise'] >= 0, gte0
assert isinstance(inputs_dict['target_policy_clip'], (float, int)) and \
    inputs_dict['target_policy_clip'] >= 0, gte0

# shared parameter tests
assert isinstance(inputs_dict['target_update_rate'], (float, int)) and \
    inputs_dict['target_update_rate'] > 0, gt0
assert inputs_dict['s_dist'] == ('N' or 'L') or \
    (inputs_dict['algo_name'][0] == 'SAC' and inputs_dict['s_dist'] == 'MVN'), \
        'must be either "N", "S" or "MVN" (only for SAC)'
assert isinstance(inputs_dict['batch_size'], dict) and \
    isinstance(inputs_dict['batch_size']['TD3'], (float, int)) and \
        int(inputs_dict['batch_size']['TD3']) >= 1 and \
            isinstance(inputs_dict['batch_size']['SAC'], (float, int)) and \
                int(inputs_dict['batch_size']['SAC']) >= 1, \
                    'mini-batch sizes must be at least 1 for all algorithms'
assert isinstance(inputs_dict['grad_step'], dict) and \
    isinstance(inputs_dict['grad_step']['TD3'], (float, int)) and \
        int(inputs_dict['grad_step']['TD3']) >= 1 and \
            isinstance(inputs_dict['grad_step']['SAC'], (float, int)) and \
                int(inputs_dict['grad_step']['SAC']) >= 1, \
                    'gradient step must be at least 1 for all algorithms'

# environment tests
assert isinstance(inputs_dict['algo_name'], list) and \
    set(inputs_dict['algo_name']).issubset(set(['SAC', 'TD3'])), \
        'algorithms must be a list of "SAC" and/or "TD3"'
assert isinstance(inputs_dict['critic_loss'], list) and \
    set(inputs_dict['critic_loss']).issubset(set(['MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'CIM', 'MSE2', 'MSE4', 'MSE6'])), \
        'critic losses must be a list of "MSE", "HUB", "MAE", "HSC", "CAU", "CIM", "MSE2", "MSE4", and/or "MSE6"'
assert isinstance(inputs_dict['multi_steps'], list) and \
    all(isinstance(mstep, int) for mstep in inputs_dict['multi_steps']) and \
        all(mstep >= 1 for mstep in inputs_dict['multi_steps']), \
            'multi-steps must be a list of positve integers'
assert isinstance(inputs_dict['ENV_KEY'], int) and \
    inputs_dict['ENV_KEY'] >= 0 and inputs_dict['ENV_KEY'] < len(gym_envs.keys()), \
         'ENV_KEY must be an integrer and match those in dictionary gym_envs'
assert isinstance(gym_envs[str(inputs_dict['ENV_KEY'])], list) and \
    isinstance(gym_envs[str(inputs_dict['ENV_KEY'])][0], str) and \
        all(isinstance(x, (float, int)) for x in gym_envs[str(inputs_dict['ENV_KEY'])][1:]), \
        'environment details must be a list of the form [string, real, real, real]'
assert int(gym_envs[str(inputs_dict['ENV_KEY'])][1]) >= 1, 'must have at least one environment state'
assert int(gym_envs[str(inputs_dict['ENV_KEY'])][2]) >= 1, 'must have at least one environment action'
assert int(gym_envs[str(inputs_dict['ENV_KEY'])][3]) >= 0 and \
    int(gym_envs[str(inputs_dict['ENV_KEY'])][3]) <= int(inputs_dict['n_cumsteps']), \
        'warm-up must be less than or equal to total training steps'

# SAC algorithm method checks
assert hasattr(Agent_sac, 'select_next_action'), 'missing SAC agent action selection'
assert hasattr(Agent_sac, 'store_transistion'), 'missing SAC transition storage functionality'
assert hasattr(Agent_sac, 'learn'), 'missing SAC agent learning functionality'
assert hasattr(Agent_sac, 'save_models'), 'missing SAC agent save functionality'
assert hasattr(Agent_sac, 'load_models'), 'missing SAC agent load functionality'
assert hasattr(ActorNetwork_sac, 'stochastic_uv'), 'missing SAC univariate sampling'
assert hasattr(ActorNetwork_sac, 'stochastic_mv_gaussian'), 'missing SAC multi-variate Gaussian sampling'
assert hasattr(ActorNetwork_sac, 'save_checkpoint'), 'missing SAC actor saving functionality'
assert hasattr(ActorNetwork_sac, 'load_checkpoint'), 'missing SAC actor load functionality'
assert hasattr(CriticNetwork_sac, 'forward'), 'missing SAC critic forward propagation'
assert hasattr(CriticNetwork_sac, 'save_checkpoint'), 'missing SAC critic saving functionality'
assert hasattr(CriticNetwork_sac, 'load_checkpoint'), 'missing SAC critic load functionality'

# TD3 algorithm method checks
assert hasattr(Agent_td3, 'select_next_action'), 'missing TD3 agent action selection'
assert hasattr(Agent_td3, 'store_transistion'), 'missing TD3 transition storage functionality'
assert hasattr(Agent_td3, 'learn'), 'missing TD3 agent learning functionality'
assert hasattr(Agent_td3, 'save_models'), 'missing TD3 agent save functionality'
assert hasattr(Agent_td3, 'load_models'), 'missing TD3 agent load functionality'
assert hasattr(ActorNetwork_td3, 'forward'), 'missing TD3 actor forward propagation'
assert hasattr(ActorNetwork_td3, 'save_checkpoint'), 'missing TD3 actor saving functionality'
assert hasattr(ActorNetwork_td3, 'load_checkpoint'), 'missing TD3 actor load functionality'
assert hasattr(CriticNetwork_td3, 'forward'), 'missing TD3 critic forward propagation'
assert hasattr(CriticNetwork_td3, 'save_checkpoint'), 'missing TD3 critic saving functionality'
assert hasattr(CriticNetwork_td3, 'load_checkpoint'), 'missing TD3 critic load functionality'

# replay buffer method checks
assert hasattr(ReplayBuffer, 'store_exp'), 'missing transition store functionality'
assert hasattr(ReplayBuffer, 'sample_exp'), 'missing uniform transition sampling functionality'

if __name__ == '__main__':

    if ENV_KEY <= 13:
        additive_env(gym_envs=gym_envs, inputs=inputs_dict)
    else:
        multiplicative_env(gym_envs=gym_envs, inputs=inputs_dict)