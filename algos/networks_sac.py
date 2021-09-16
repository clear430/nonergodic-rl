import extras.utils as utils 
import os
import torch as T
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple

class ActorNetwork(nn.Module):
    """
    Actor network for single GPU.

    Methods:
        forward(state):
            Forward propogate state to obtain policy distribution moments for 
            each action component.

        stochastic_uv(state):
            Obtain tanh bounded actions values and log probabilites for a state 
            using stochastic policy.

        stochasic_mv_gaussian(state):
            Obtain tanh bounded actions values and log probabilites for a state 
            using multi-variable spherical Gaussian distribution with each state 
            having a unique covariance matrix. 

        save_checkpoint():
            Saves network parameters.
            
        load_checkpoint():
            Loads network parameters.
    """

    def __init__(self, inputs_dict: dict, target: bool):
        """
        Intialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            inputs_dict: dictionary containing all execution details
            target: whether constructing target network (1) or not (0)
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = sum(inputs_dict['input_dims'])
        self.num_actions = int(inputs_dict['num_actions'])
        self.max_action = float(inputs_dict['max_action'])

        nn_name = 'actor' if target == 0 else 'actor_target'
        
        fc1_dim = int(inputs_dict['sac_layer_1_units'])
        fc2_dim = int(inputs_dict['sac_layer_2_units'])
        lr_alpha = inputs_dict['sac_actor_learn_rate']
        self.stoch = str(inputs_dict['s_dist'])
        self.reparam_noise = inputs_dict['reparam_noise']
        
        # directory to save network checkpoints
        dir = './models/'
        dir += 'additive/' if inputs_dict['dynamics'] == 'A' else 'multiplicative/'
        dir += str(inputs_dict['env_id'])

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        file = utils.save_directory(inputs_dict, results=False) + '_' + nn_name + '.pt'
        self.file_checkpoint = os.path.join(file)

        # network inputs environment state space features
        self.fc1 = nn.Linear(self.input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.pi = nn.Linear(fc2_dim, self.num_actions * 2)

        self.optimiser = optim.Adam(self.parameters(), lr=lr_alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.cuda.FloatTensor) -> T.cuda.FloatTensor:
        """
        Forward propogation of state to obtain fixed Gaussian distribution parameters
        (moments) for each possible action component. 

        Parameters:
            state: current environment state

        Returns:
            moments: first half columns for deterministic loc of action components
                     and second half columns for log scales of action components
        """
        actions_2x = self.fc1(state)
        actions_2x = F.relu(actions_2x)
        actions_2x = self.fc2(actions_2x)
        actions_2x = F.relu(actions_2x)
        moments = self.pi(actions_2x)

        return moments
    
    def stochastic_uv(self, state: T.cuda.FloatTensor) \
            -> Tuple[T.cuda.FloatTensor, T.cuda.FloatTensor]:
        """ 
        Stochastic action selection sampled from several unbounded univarite distirbution
        using the reparameterisation trick from https://arxiv.org/pdf/1312.6114.pdf. Addition
        of constant reparameterisation noise to the logarithm is crucial, as verified in both
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py
        https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/SAC/networks.py
        where orders of magnitude smaller than 1e-6 prevent learning from occuring. 
        
        Parameters:
            state: current environment state or mini-batch

        Returns:
            bounded_action: action truncated by tanh and scaled by max action
            bounded_logprob_action: log probability (log likelihood) of sampled truncated action 
        """
        moments = self.forward(state)
        mu, log_scale = moments[:, :self.num_actions], moments[:, self.num_actions:]
        scale = log_scale.exp()
        
        if self.stoch == 'N':
            probabilities = Normal(loc=mu, scale=scale)
        else:
            probabilities = Laplace(loc=mu, scale=scale)

        # reparmeterise trick for random variable sample to be pathwise differentiable
        unbounded_action = probabilities.rsample().to(self.device)
        bounded_action = T.tanh(unbounded_action) * self.max_action
        unbounded_logprob_action = probabilities.log_prob(unbounded_action) \
                                   .sum(1, keepdim=True).to(self.device)
        
        # ensure defined bounded log by adding minute noise
        log_inv_jacobian = T.log(1 - (bounded_action / self.max_action)**2 
                                 + self.reparam_noise).sum(dim=1, keepdim=True)
        bounded_logprob_action = unbounded_logprob_action - log_inv_jacobian

        return bounded_action, bounded_logprob_action
    
    def stochastic_mv_gaussian(self, state: T.cuda.FloatTensor) \
            -> Tuple[T.cuda.FloatTensor, T.cuda.FloatTensor]:
        """
        Stochastic action selection sampled from unbounded spherical Gaussian input 
        noise with tanh bounding using Jacobian transformation. Allows each mini-batch 
        state to have a unique covariance matrix allowing faster learning in terms of 
        cumulative steps but significantly longer run time per step. Likely only feasible
        if PyTorch implements a multivariate normal sampling distirbution.

        Parameters:
            state: current environment state or mini-batch

        Returns:
            bounded_action: action truncated by tanh and scaled by max action
            bounded_logprob_action: log probability of sampled truncated action 
        """
        moments = self.forward(state)
        batch_size = moments.size()[0]
        mu, log_var = moments[:, :self.num_actions], moments[:, self.num_actions:]
        var = log_var.exp()

        if batch_size > 1:
            pass
        else:
            mu, var = mu.view(-1), var.view(-1)
        
        # create diagonal covariance matrices for each sample and perform Cholesky decomposition
        cov_mat = T.stack([T.eye(self.num_actions) for i in range(batch_size)]).to(self.device)

        if batch_size > 1:
            for sample in range(batch_size):
                for vol in range(self.num_actions):
                        cov_mat[sample, vol, vol] = var[sample, vol]    # diagonal elements are variance
        else:
            for vol in range(self.num_actions):     
                cov_mat[0, vol, vol] = var[vol]

        chol_ltm = T.linalg.cholesky(cov_mat)

        probabilities = MultivariateNormal(loc=mu, scale_tril=chol_ltm)

        # reparmeterise trick for random variable sample to be pathwise differentiable
        unbounded_action = probabilities.rsample().to(self.device)
        bounded_action = T.tanh(unbounded_action) * self.max_action
        unbounded_logprob_action = probabilities.log_prob(unbounded_action).to(self.device)

        # ensure defined bounded log by adding minute noise
        log_inv_jacobian = T.log(1 - (bounded_action / self.max_action)**2 
                                 + self.reparam_noise).sum(dim=1)
        bounded_logprob_action = unbounded_logprob_action - log_inv_jacobian

        return bounded_action, bounded_logprob_action

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        print('Loading actor checkpoint')
        self.load_state_dict(T.load(self.file_checkpoint))

class CriticNetwork(nn.Module):
    """
    Critic network for single GPU. 

    Methods:
        forward(state):
            Forward propogate concatenated state and action to obtain soft Q-values.

        save_checkpoint():
            Saves network parameters.
            
        load_checkpoint():
            Loads network parameters.
    """

    def __init__(self, inputs_dict: dict, critic: int, target: bool):
        """
        Intialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            inputs_dict: dictionary containing all execution details
            critic: number assigned to critic
            target: whether constructing target network (1) or not (0)
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = sum(inputs_dict['input_dims'])
        self.num_actions = int(inputs_dict['num_actions'])
        self.max_action = float(inputs_dict['max_action'])

        nn_name = 'critic' if target == 0 else 'target_critic'
        nn_name += '_'+str(critic)
        
        fc1_dim = int(inputs_dict['sac_layer_1_units'])
        fc2_dim = int(inputs_dict['sac_layer_2_units'])
        lr_beta = inputs_dict['sac_critic_learn_rate']

        # directory to save network checkpoints
        dir = './models/'
        dir += 'additive/' if inputs_dict['dynamics'] == 'A' else 'multiplicative/'
        dir += str(inputs_dict['env_id'])

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        file = utils.save_directory(inputs_dict, results=False) + '_' + nn_name + '.pt'
        self.file_checkpoint = os.path.join(file)

        # network inputs environment state space features and number of actions
        self.fc1 = nn.Linear(self.input_dims + self.num_actions, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=lr_beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.cuda.FloatTensor, action: T.cuda.FloatTensor) \
            -> T.cuda.FloatTensor:
        """
        Forward propogation of state-action pair to obtain soft Q-value.

        Parameters:
            state: current environment state
            action: continuous actions taken to arrive at current state          

        Returns:
            soft_Q (float): estimated soft Q action-value
        """
        Q_action_value = self.fc1(T.cat([state, action], dim=1))
        Q_action_value = F.relu(Q_action_value)
        Q_action_value = self.fc2(Q_action_value)
        Q_action_value = F.relu(Q_action_value)
        soft_Q = self.q(Q_action_value)

        return soft_Q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_checkpoint))