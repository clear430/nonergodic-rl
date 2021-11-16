#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  algo_td3.py
python version:         3.9
torch verison:          1.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for executing the Twin Delayed DDPG (TD3) algorithm.
"""

import numpy as np
import os
import torch as T
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from typing import NoReturn, Tuple

from algos.networks_td3 import ActorNetwork, CriticNetwork
from extras.replay import ReplayBuffer
from extras.replay_torch import ReplayBufferTorch
import extras.critic_loss as closs
import extras.utils as utils

class Agent_td3():
    """
    DDPG policy optimisation based on https://arxiv.org/pdf/1509.02971.pdf.
    TD3 agent algorithm based on https://arxiv.org/pdf/1802.09477.pdf.
    Agent learning using PyTorch neural networks described in
    https://proceedings.neurips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

    Methods:
    store_transistion(state, action, reward, next_state, done):
        Stores current step details into replay buffer.

    select_next_action(state):
        Select next agent action based on provided single state using given 
        deterministic policy distribution.

    eval_next_action(state):
        Select next agent action for evaluation based on provided state using given 
        deterministic policy distribution with zero noise.

    _mini_batch(batch_size):
        Randomly collects mini-batch from replay buffer and sends to GPU.

    _multi_step_target(batch_rewards, batch_next_states, batch_dones, batch_epis_rewards):
        Predicits next actions from next states in mini-batch, forward propagates 
        these state-action pairs, and construct bootstrapped target Q-values.

    learn():
        Perform agent learning by sequentially minimising critic and policy loss 
        functions every step. Updates Cauchy scale and CIM kernel size parameters.

    _update_actor_parameters(tau):
        Update actor target network parameters with Polyak averaging.

    _update_critic_parameters(tau):
        Update critics target network parameters with Polyak averaging.

    save_models():
        Save actor-critic network parameters.

    load_models():
        Load actor-critic network parameters.
    """

    def __init__(self, env: object, inputs_dict: dict):
        """
        Intialise actor-critic networks and experience replay buffer.

        Parameters:
            env: gym environment
            inputs_dict: dictionary containing all execution details
        """
        self.env = env
        self.input_dims = sum(inputs_dict['input_dims'])    # input dimensions tuple
        self.num_actions = int(inputs_dict['num_actions'])
        self.max_action = float(inputs_dict['max_action'])
        self.min_action = float(inputs_dict['min_action'])

        self.actor_update_interval = int(inputs_dict['td3_actor_step_update'])
        self.target_actor_update = int(inputs_dict['td3_target_actor_update'])
        self.target_critic_update = int(inputs_dict['td3_target_critic_update'])
        self.tau = inputs_dict['target_update_rate']
        self.stoch = str(inputs_dict['s_dist'])
        self.batch_size = int(inputs_dict['batch_size'][inputs_dict['algo']])

        # scale standard deviations by max continous action
        self.policy_noise = inputs_dict['policy_noise'] * self.max_action
        self.target_policy_noise = inputs_dict['target_policy_noise'] * self.max_action
        self.target_policy_clip = inputs_dict['target_policy_clip'] * self.max_action
    
        # convert Gaussian standard deviation to Laplace diveristy
        self.policy_scale = np.sqrt(self.policy_noise**2 / 2)
        self.target_policy_scale = np.sqrt(self.target_policy_noise**2 / 2)

        self.buffer_torch = inputs_dict['buffer_gpu']
        if self.buffer_torch == False:
            self.memory = ReplayBuffer(inputs_dict)
        else:
            self.memory = ReplayBufferTorch(inputs_dict)

        self.gamma = inputs_dict['discount']
        self.multi_steps = int(inputs_dict['multi_steps'])
        self.cauchy_scale_1 = inputs_dict['cauchy_scale']
        self.cauchy_scale_2 = inputs_dict['cauchy_scale']
        
        self.actor_percentile = inputs_dict['actor_percentile']
        self.actor_bottom_count = int(self.actor_percentile * self.batch_size)

        self.warmup = int(inputs_dict['random'])      
        self.loss_type = str(inputs_dict['loss_fn'])

        self.time_step = 0
        self.learn_step_cntr = 0

        # directory to save network checkpoints
        dir = './models/'
        if inputs_dict['dynamics'] == 'A':
            dir += 'additive/' 
        elif inputs_dict['dynamics'] == 'M':
            dir +=  'multiplicative/'
        else:
            dir += 'market/'
        
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir += str(inputs_dict['env_id'])
        
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        # generic directory and name of PyTorch model
        model_name = utils.save_directory(inputs_dict, results=False)

        self.actor = ActorNetwork(inputs_dict, model_name, target=0)
        self.target_actor = ActorNetwork(inputs_dict, model_name, target=1)
        self.critic_1 = CriticNetwork(inputs_dict, model_name, critic=1, target=0)
        self.target_critic_1 = CriticNetwork(inputs_dict, model_name, critic=1, target=1)
        self.critic_2 = CriticNetwork(inputs_dict, model_name, critic=2, target=0) 
        self.target_critic_2 = CriticNetwork(inputs_dict, model_name, critic=2, target=1)

        if self.stoch == 'N':
            self.pdf = Normal(loc=0, scale=self.policy_noise)
            self.target_pdf = Normal(loc=0, scale=self.target_policy_noise)
        elif self.stoch == 'L':
            self.pdf = Laplace(loc=0, scale=self.policy_scale)
            self.target_pdf = Laplace(loc=0, scale=self.target_policy_scale)
        
        self.critic_mean = str(inputs_dict['critic_mean_type'])
        self.shadow_low_mul = inputs_dict['shadow_low_mul']
        self.shadow_high_mul = inputs_dict['shadow_high_mul']

        # intialisation for tail exponent estimation
        self.zipf_x = (T.ones((self.batch_size,)) + self.batch_size).view(-1)
        for x in range(self.batch_size):
            self.zipf_x[x] = self.zipf_x[x] / (x + 1)
        self.zipf_x = T.log(self.zipf_x)
        self.zipf_x = (self.zipf_x - T.mean(self.zipf_x)).to(self.critic_1.device)
        self.zipf_x2 = T.sum(self.zipf_x**2).to(self.critic_1.device)

    def store_transistion(self, state: np.ndarray, action: np.ndarray, reward: float, 
                          next_state: np.ndarray, done: bool) -> NoReturn:
        """
        Store a transistion to the buffer containing a total up to max_size.

        Parameters:
            state: current environment state
            action: continuous actions taken to arrive at current state
            reward: reward from current environment state
            next_state: next environment state
            done: flag if current state is terminal
        """
        self.memory.store_exp(state, action, reward, next_state, done)

    def select_next_action(self, state: T.FloatTensor) -> Tuple[np.ndarray, T.FloatTensor]:
        """
        Agent selects next action from determinstic policy with noise added to each component, 
        or during warmup a random action taken.

        Parameters:
            state: current environment state

        Return:
            numpy_next_action: action to be taken by agent in next step for gym
            next_action: action to be taken by agent in next step
        """
        if self.time_step >= self.warmup:
        
            current_state = T.tensor(state, dtype=T.float).to(self.actor.device)
            action_noise = self.pdf.sample(sample_shape=(self.num_actions,)).to(self.actor.device)
            mu = action_noise + self.actor.forward(current_state)

            next_action = T.clamp(mu, self.min_action, self.max_action)
            numpy_next_action = next_action.detach().cpu().numpy()

            return numpy_next_action, next_action

        else:
            numpy_next_action = self.env.action_space.sample()
            next_action = None

        self.time_step += 1
        
        return numpy_next_action, next_action

    def eval_next_action(self, state: T.FloatTensor) -> np.ndarray:
        """
        Agent selects next action from determinstic policy with no noise used for 
        direct agent inference/evaluation.

        Parameters:
            state: current environment state

        Return:
            numpy_next_action: action to be taken by agent in next step for gym
        """
        current_state = T.tensor(state, dtype=T.float).to(self.actor.device)
        
        return self.actor.forward(current_state).detach().cpu().numpy()

    def _mini_batch(self) -> Tuple[T.FloatTensor, T.FloatTensor, T.FloatTensor, 
                                   T.FloatTensor, T.BoolTensor, T.IntTensor]:
        """
        Uniform sampling from replay buffer and send to GPU.

        Returns:
            states: batch of environment states
            actions: batch of continuous actions taken to arrive at states
            rewards: batch of (discounted multi-step) rewards from current states
            next_states: batch of next environment states
            dones (bool): batch of done flags
            eff_length: batch of effective multi-step episode lengths
        """
        if self.memory.mem_idx < self.batch_size:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        batch_states, batch_actions, batch_rewards, \
            batch_next_states, batch_dones, batch_eff_length = self.memory.sample_exp()

        if self.buffer_torch == False:
            batch_states = T.tensor(batch_states, dtype=T.float, device=self.critic_1.device)
            batch_actions = T.tensor(batch_actions, dtype=T.float, device=self.critic_1.device)
            batch_rewards = T.tensor(batch_rewards, dtype=T.float, device=self.critic_1.device)
            batch_next_states = T.tensor(batch_next_states, dtype=T.float, device=self.critic_1.device)
            batch_dones = T.tensor(batch_dones, dtype=T.bool, device=self.critic_1.device)
            batch_eff_length = T.tensor(batch_eff_length, dtype=T.int, device=self.critic_1.device)

        return batch_states, batch_actions, batch_rewards, batch_next_states, \
               batch_dones, batch_eff_length 

    def _multi_step_target(self, batch_rewards: T.FloatTensor, batch_next_states: T.FloatTensor, 
                           batch_dones: T.BoolTensor, batch_eff_length: T.IntTensor) \
            -> T.FloatTensor:
        """
        Multi-step target Q-values for mini-batch with regularisation through noise addition. 

        Parameters:
            batch_rewards: batch of (discounted multi-step) rewards from current states
            batch_next_states: batch of next environment states
            batch_dones: batch of done flags
            batch_eff_length: batch of effective multi-step episode lengths
        
        Returns:
            batch_target: clipped double multi-step target Q-values
        """
        # ensure memory buffer large enough for mini-batch
        if self.memory.mem_idx <= self.batch_size: 
            return np.nan
        
        # add random noise to each component of next target action with clipping     
        target_action_noise = self.target_pdf.sample((self.batch_size, self.num_actions)).to(self.actor.device)
        target_action_noise = target_action_noise.clamp(-self.target_policy_clip, self.target_policy_clip)

        # predict next agent action
        batch_next_actions = self.target_actor.forward(batch_next_states)
        batch_next_actions = (batch_next_actions + target_action_noise).clamp(self.min_action, self.max_action)

        # obtain twin target Q-values for mini-batch and check terminal status
        q1_target = self.target_critic_1.forward(batch_next_states, batch_next_actions).view(-1)
        q2_target = self.target_critic_2.forward(batch_next_states, batch_next_actions).view(-1)
        q1_target[batch_dones], q2_target[batch_dones] = 0.0, 0.0

        # clipped double target critic values with bootstrapping
        q_target = T.min(q1_target, q2_target)
        target = batch_rewards + self.gamma**batch_eff_length * q_target
        batch_target = target.view(self.batch_size, 1)

        return batch_target

    def learn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                             np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Agent learning via TD3 algorithm with multi-step bootstrapping and robust critic loss.

        Returns:
            loss: empirical mean / min / max /shadow mean of critic losses, critic tail exponents, mean actor loss
            loss_params: list of Cauchy scale parameters and CIM kernel sizes for twin critics
        """
        # return nothing till batch size less than replay buffer
        if self.memory.mem_idx <= self.batch_size:
            loss = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            loss_params = [np.nan, np.nan, np.nan, np.nan]
            return loss, np.nan, loss_params
        
        # uniformly sample from replay buffer (off-policy) 
        batch_states, batch_actions, batch_rewards, \
            batch_next_states, batch_dones, batch_eff_length = self._mini_batch()

        # estimate target critic value
        batch_target = self._multi_step_target(batch_rewards, batch_next_states, batch_dones, batch_eff_length)

        # obtain twin current state Q-values for current step
        q1 = self.critic_1.forward(batch_states, batch_actions).view(-1)
        q2 = self.critic_2.forward(batch_states, batch_actions).view(-1)
        q1, q2 = q1.view(self.batch_size, 1), q2.view(self.batch_size, 1)

        # updates CIM kernel size empirically
        kernel_1 = closs.cim_size(q1, batch_target).cpu().numpy()
        kernel_2 = closs.cim_size(q2, batch_target).cpu().numpy()

        # backpropogation of critic loss
        self.critic_1.optimiser.zero_grad(set_to_none=True)
        self.critic_2.optimiser.zero_grad(set_to_none=True)

        q1_mean, q1_min, q1_max, \
            q1_shadow, q1_alpha = closs.loss_function(q1, batch_target, self.shadow_low_mul, self.shadow_high_mul,
                                                       self.zipf_x, self.zipf_x2, self.loss_type, 
                                                       self.cauchy_scale_1, kernel_1)
        q2_mean, q2_min, q2_max, \
            q2_shadow, q2_alpha = closs.loss_function(q2, batch_target, self.shadow_low_mul, self.shadow_high_mul,
                                                      self.zipf_x, self.zipf_x2, self.loss_type, 
                                                      self.cauchy_scale_2, kernel_2)

        # ensure consistent mean selection for learning
        if self.critic_mean == 'E':
            q1_loss, q2_loss = q1_mean, q2_mean
        else:
            q1_loss, q2_loss = q1_shadow, q2_shadow

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        # updates Cauchy scale parameter using the Nagy algorithm
        self.cauchy_scale_1 = closs.nagy_algo(q1, batch_target, self.cauchy_scale_1).cpu().numpy()
        self.cauchy_scale_2 = closs.nagy_algo(q2, batch_target, self.cauchy_scale_2).cpu().numpy()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.target_critic_update == 0:        
            self._update_critic_parameters()

        cpu_q1_mean, cpu_q2_mean = q1_mean.detach().cpu().numpy(), q2_mean.detach().cpu().numpy()
        cpu_q1_min, cpu_q2_min = q1_min.detach().cpu().numpy(), q2_min.detach().cpu().numpy()
        cpu_q1_max, cpu_q2_max = q1_max.detach().cpu().numpy(), q2_max.detach().cpu().numpy()
        cpu_q1_shadow, cpu_q2_shadow = q1_shadow.detach().cpu().numpy(), q2_shadow.detach().cpu().numpy()
        cpu_q1_alpha, cpu_q2_alpha = q1_alpha.detach().cpu().numpy(), q2_alpha.detach().cpu().numpy()

        loss = [cpu_q1_mean, cpu_q2_mean, cpu_q1_min, cpu_q2_min, cpu_q1_max, cpu_q2_max, 
                cpu_q1_shadow, cpu_q2_shadow, cpu_q1_alpha, cpu_q2_alpha, np.nan]
        loss_params = [self.cauchy_scale_1, self.cauchy_scale_2, kernel_1, kernel_2]

        if self.learn_step_cntr % self.actor_update_interval != 0:
            return loss, np.nan, loss_params

        # deterministic policy gradient ascent approximation
        self.actor.optimiser.zero_grad(set_to_none=True)
        batch_next_actions = self.actor.forward(batch_states)
        actor_q1_loss = self.critic_1.forward(batch_states, batch_next_actions).view(-1)

        # application of fractional Kelly betting where emphasis is placed on improving the worst peformers
        if self.actor_percentile != 1:
            actor_q1_loss = actor_q1_loss.sort(descending=False)[0]
            actor_q1_loss = actor_q1_loss[:self.actor_bottom_count]

        actor_q1_loss = -T.mean(actor_q1_loss)
        actor_q1_loss.backward()
        
        self.actor.optimiser.step()

        if self.learn_step_cntr % self.target_actor_update == 0:        
            self._update_actor_parameters()
        
        cpu_actor_loss = actor_q1_loss.detach().cpu().numpy()
        loss[-1] = cpu_actor_loss

        return loss, np.nan, loss_params

    def _update_actor_parameters(self) -> NoReturn:
        """
        Update target actor deep network parameters with Polyak averaging rate smoothing.
        """
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)       

    def _update_critic_parameters(self) -> NoReturn:
        """
        Update target critic deep network parameters with Polyak averaging rate smoothing.
        """
        for param_1, target_param_1, param_2, target_param_2 in \
          zip(self.critic_1.parameters(), self.target_critic_1.parameters(), 
              self.critic_2.parameters(), self.target_critic_2.parameters()):
   
            target_param_1.data.copy_(self.tau * param_1.data + (1 - self.tau) * target_param_1.data)
            target_param_2.data.copy_(self.tau * param_2.data + (1 - self.tau) * target_param_2.data)

    def save_models(self) -> NoReturn:
        """
        Saves all 3 networks.
        """
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self) -> NoReturn:
        """
        Loads all 3 networks.
        """
        print('Loading network checkpoints')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()