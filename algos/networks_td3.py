#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  networks_td3.py
python version:         3.9
torch verison:          1.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for generating actor-critic deep (linear) neural networks for the 
    Twin Delayed DDPG (TD3) algorithm.
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorNetwork(nn.Module):
    """
    Actor network for single GPU.

    Methods:
        forward(state):
            Forward propogate states to obtain next action components.

        save_checkpoint():
            Saves network parameters.
            
        load_checkpoint():
            Loads network parameters.
    """

    def __init__(self, inputs_dict: dict, model_name: str, target: bool):
        """
        Intialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            inputs_dict: dictionary containing all execution details
            model_name: directory and naming of model
            target: whether constructing target network (1) or not (0)
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = sum(inputs_dict['input_dims'])
        self.num_actions = int(inputs_dict['num_actions'])
        self.max_action = float(inputs_dict['max_action'])

        nn_name = 'actor' if target == 0 else 'actor_target'
        
        fc1_dim = int(inputs_dict['td3_layer_1_units'])
        fc2_dim = int(inputs_dict['td3_layer_2_units'])
        lr_alpha = inputs_dict['td3_actor_learn_rate']
            
        file = model_name + '_' + nn_name + '.pt'
        self.file_checkpoint = os.path.join(file)

        # network inputs environment state space features
        self.fc1 = T.jit.script(nn.Linear(self.input_dims, fc1_dim))
        self.fc2 = T.jit.script(nn.Linear(fc1_dim, fc2_dim))
        self.mu = T.jit.script(nn.Linear(fc2_dim, self.num_actions))

        self.optimiser = optim.Adam(self.parameters(), lr=lr_alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.FloatTensor) -> T.FloatTensor:
        """
        Forward propogation of mini-batch states to obtain next actor actions.

        Parameters:
            state: current environment states

        Returns:
            actions_scaled: next agent actions between -1 and 1 scaled by max action
        """
        actions = self.fc1(state)
        actions = F.relu(actions)
        actions = self.fc2(actions)
        actions = F.relu(actions)
        actions_scaled = T.tanh(self.mu(actions)) * self.max_action

        return actions_scaled

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_checkpoint))

class CriticNetwork(nn.Module):
    """
    Critic network for single GPU. 

    Methods:
        forward(state):
            Forward propogate concatenated state and action to obtain Q-values.

        save_checkpoint():
            Saves network parameters.
            
        load_checkpoint():
            Loads network parameters.
    """
    
    def __init__(self, inputs_dict: dict, model_name:str, critic: int, target: bool):
        """
        Intialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            inputs_dict: dictionary containing all execution details
            model_name: directory and naming of model
            critic: number assigned to critic
            target: whether constructing target network (1) or not (0)
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = sum(inputs_dict['input_dims'])
        self.num_actions = int(inputs_dict['num_actions'])
        self.max_action = float(inputs_dict['max_action'])

        nn_name = 'critic' if target == 0 else 'target_critic'
        nn_name += '_'+str(critic)
        
        fc1_dim = int(inputs_dict['td3_layer_1_units'])
        fc2_dim = int(inputs_dict['td3_layer_2_units'])
        lr_beta = inputs_dict['td3_critic_learn_rate']
        
        file = model_name + '_' + nn_name + '.pt'
        self.file_checkpoint = os.path.join(file)

        # network inputs environment state space features and number of actions
        self.fc1 = T.jit.script(nn.Linear(self.input_dims + self.num_actions, fc1_dim))
        self.fc2 = T.jit.script(nn.Linear(fc1_dim, fc2_dim))
        self.q = T.jit.script(nn.Linear(fc2_dim, 1))

        self.optimiser = optim.Adam(self.parameters(), lr=lr_beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.FloatTensor, action: T.FloatTensor) \
            -> T.FloatTensor:
        """
        Forward propogation of mini-batch state-action pairs to obtain Q-value.

        Parameters:
            state: current environment states
            action: continuous next actions taken at current states      

        Returns:
            Q (float): estimated Q action-value
        """
        Q_action_value = self.fc1(T.cat([state, action], dim=1))
        Q_action_value = F.relu(Q_action_value)
        Q_action_value = self.fc2(Q_action_value)
        Q_action_value = F.relu(Q_action_value)
        Q = self.q(Q_action_value)

        return Q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_checkpoint))