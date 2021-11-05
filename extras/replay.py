#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  replay.py
python version:         3.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Responsible for creating experience replay buffer with uniform step sampling. 
"""

import numpy as np
from typing import List, Tuple

class ReplayBuffer():
    """
    Experience replay buffer with uniform sampling based on 
    https://link.springer.com/content/pdf/10.1023%2FA%3A1022628806385.pdf.

    Methods:
        store_exp(state, action, reward, next_state, done):
            Stores current step into experience replay buffer.

        _construct_history(step, epis_history):
            Generate reward and state-action pair history for a sampled step.

        _episode_rewards_states_actions(batch):
            Collect reward and state-action pair histories for the batch.

        _multi_step_rewards_states_actions(reward_history, state_history, action_history, multi_length):
            Generate multi-step rewards and intial state-action pair for a sampled step.

        _multi_step_batch(step_rewards, step_states, step_actions):
            Collect multi-step rewards and intial state-action pairs for the batch.
        
        sample_exp():
            Uniformly sample mini-batch from experience replay buffer.
    """

    def __init__(self, inputs_dict: dict):
        """
        Intialise class varaibles by creating empty numpy buffer arrays.

        Paramters:
            inputs_dict: dictionary containing all execution details
        """
        self.input_dims = sum(inputs_dict['input_dims'])
        self.num_actions = int(inputs_dict['num_actions'])
        self.batch_size = int(inputs_dict['batch_size'][inputs_dict['algo']])
        self.gamma = inputs_dict['discount']
        self.multi_steps = int(inputs_dict['multi_steps'])
        self.r_abs_zero = inputs_dict['r_abs_zero']

        self.dyna = str(inputs_dict['dynamics'])
        
        if int(inputs_dict['buffer']) <= int(inputs_dict['n_cumsteps']):
            self.mem_size = int(inputs_dict['buffer'])
        else:
            self.mem_size = int(inputs_dict['n_cumsteps'])

        self.mem_idx = 0

        self.state_memory = np.zeros((self.mem_size, self.input_dims))
        self.action_memory = np.zeros((self.mem_size, self.num_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.next_state_memory = np.zeros((self.mem_size, self.input_dims))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

        self.epis_idx = [np.nan]
        self.epis_reward_memory = []
        self.epis_state_memory = []
        self.epis_action_memory = []

    def store_exp(self, state: np.ndarray, action: np.ndarray, reward: float, 
                  next_state: np.ndarray, done: bool):
        """
        Store a transistion to the buffer containing a total up to a maximum size and log 
        history of rewards, states, and actions for each episode.

        Parameters:
            state: current environment state
            action: continuous actions taken to arrive at current state
            reward: reward from arriving at current environment state
            next_state: next or new environment state
            done: flag if new state is terminal
        """
        idx = self.mem_idx % self.mem_size

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = np.max(reward, self.r_abs_zero)
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = done

        # note this only works if buffer >= cumulative training steps
        if self.multi_steps > 1:
            self.epis_idx[-1] = idx

            # aggregate the reward, state, and action histories of the currently trained episode
            try:
                current_start = self.epis_idx[-2] + 1
                current_reward_memory = self.reward_memory[current_start:idx + 1]
                current_state_memory = self.next_state_memory[current_start:idx + 1]
                current_action_memory = self.action_memory[current_start:idx + 1]
            except:
                try:
                    # used for the start of a new training episode (excluding the first)
                    current_reward_memory = self.reward_memory[0:idx + 1]
                    current_state_memory = self.next_state_memory[0:idx + 1]
                    current_action_memory = self.action_memory[0:idx + 1]
                except:
                    # used for the very first training step of the first episode
                    current_reward_memory = self.reward_memory[idx]
                    current_state_memory = self.next_state_memory[idx]
                    current_action_memory = self.action_memory[idx]

            #  log history for the very first training episode
            if np.all(self.terminal_memory != True) and done is not True:
                self.epis_idx = [idx + 1]
                self.epis_reward_memory = [current_reward_memory]
                self.epis_state_memory = [current_state_memory]
                self.epis_action_memory = [current_action_memory]
 
            # reset history once the very first training episode has concluded
            if (self.terminal_memory == True).sum() == 1 and done is True:
                self.epis_idx = [idx]
                self.epis_reward_memory = []
                self.epis_state_memory = []
                self.epis_action_memory = []

            # log the aggregated history upon termination of all subsequent training episode
            if done is True:
                self.epis_idx.append(idx + 1)
                self.epis_reward_memory.append(current_reward_memory)
                self.epis_state_memory.append(current_state_memory)
                self.epis_action_memory.append(current_action_memory)

        self.mem_idx += 1

    def _contruct_history(self, step: int, epis_history: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given a single mini-batch sample (or step), obtain the history of rewards, and 
        state-action pairs.

        Parameters:
            step: index or step of sample step
            epis_history: index list containing final steps of all training epsiodes

        Returns:
            rewards: sample reward history
            states: sample state history
            actions: sample action history
        """
        # find which episode the sampled step (experience) is located
        try:
            sample_idx = int(np.max(np.where(step - epis_history > 0)) + 1) if step > epis_history[0] \
                                                                            else 0
            n_rewards = step - epis_history[sample_idx - 1] if step > epis_history[0] \
                                                            else step
        except:
            # required for intialisation 
            sample_idx, n_rewards = 0, 0

        # generate history of episode up till the sample step
        try:
            rewards = self.epis_reward_memory[sample_idx][0:n_rewards + 1]
            states = self.epis_state_memory[sample_idx][0:n_rewards + 1]
            actions = self.epis_action_memory[sample_idx][0:n_rewards + 1]
        except:
            try:
                # used for the first training episode
                rewards = self.epis_reward_memory[0][0:n_rewards + 1]
                states = self.epis_state_memory[0][0:n_rewards + 1]
                actions = self.epis_action_memory[0][0:n_rewards + 1]
            except:
                try:
                    # used for the very first training step of the first training episode
                    rewards = self.epis_reward_memory[0][0]
                    states = self.epis_state_memory[0][0]
                    actions = self.epis_action_memory[0][0]
                except:
                    # required for intialisation
                    rewards, states, actions = 0, 0, 0
            
        return rewards, states, actions

    def _episode_rewards_states_actions(self, batch: List[int]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect respective histories for each sample in the mini-batch.

        Parameters:
            batch: list of mini-batch steps representing samples
        
        Returns:
            sample_epis_rewards: mini-batch reward history
            sample_epis_states: mini-batch state history
            sample_epis_actions: mini-batch actions history
        """
        epis_history = np.array(self.epis_idx)
        batch_histories = [self._contruct_history(step, epis_history) for step in batch]

        sample_epis_rewards = [x[0] for x in batch_histories]
        sample_epis_states = [x[1] for x in batch_histories]
        sample_epis_actions = [x[2] for x in batch_histories]

        return sample_epis_rewards, sample_epis_states, sample_epis_actions

    def _multi_step_rewards_states_actions(self, reward_history: np.ndarray, state_history: np.ndarray, 
                                           action_history: np.ndarray, multi_length: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For a single mini-batch sample, generate multi-step rewards and identify intial state-action pair.

        Parameters:
            reward_history: entire reward history of sample
            state_history: entire state history of sample
            action_history: entire action history of sample
            multi_length: minimum of length of episode or multi-steps

        Returns:
            multi_reward: discounted sum of multi-step rewards
            intial_state: array of intial state before bootstrapping
            intial_action: array of intial action before bootstrapping
        """
        idx = int(multi_length)

        # the sampled step is treated as the (n-1)th step 
        discounted_rewards = [self.gamma**t * reward_history[-idx + t] for t in range(idx - 1)]

        if self.dyna == 'A':
            multi_reward = np.sum(discounted_rewards)
        else:
            multi_reward = np.prod(discounted_rewards)

        intial_state = state_history[-idx]
        intial_action = action_history[-idx]

        return multi_reward, intial_state, intial_action
    
    def _multi_step_batch(self, step_rewards: List[np.ndarray], 
                          step_states: List[np.ndarray], step_actions: List[np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect respective multi-step returns and intial state-action pairs for each sample in the mini-batch.

        Parameters:
            step_rewards: complete reward history of entire mini-batch
            step_states: complete state history of entire mini-batch
            step_actions: complete action history of entire mini-batch

        Returns:
            batch_multi_reward: discounted sum of multi-step rewards
            batch_intial_state: array of intial state before bootstrapping
            batch_intial_action: array of intial action before bootstrapping
            batch_eff_length: effective n-step bootstrapping length
        """
        # effective length taken to be the minimum of either history length or multi-steps
        batch_eff_length = np.minimum(np.array([x.shape[0] for x in step_rewards]), self.multi_steps)

        batch_multi = [self._multi_step_rewards_states_actions(step_rewards[x], step_states[x], 
                                                               step_actions[x], batch_eff_length[x]) 
                       for x in range(self.batch_size)]

        batch_multi_rewards = np.array([batch_multi[x][0] for x in range(self.batch_size)])
        batch_states = np.array([batch_multi[x][1] for x in range(self.batch_size)])
        batch_actions = np.array([batch_multi[x][2] for x in range(self.batch_size)])
        
        return batch_multi_rewards, batch_states, batch_actions, batch_eff_length

    def sample_exp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniformly sample a batch from replay buffer for agent learning

        Returns:
            states: batch of environment states
            actions: batch of continuous actions taken to arrive at states
            rewards: batch of (discounted multi-step)  rewards from current states
            next_states: batch of next environment states
            dones (bool): batch of done flags
            batch_eff_length: batch of effective multi-step episode lengths
        """
        # pool batch from either partial or fully populated buffer
        max_mem = min(self.mem_idx, self.mem_size)
        batch = np.random.choice(max_mem, size=self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        batch_eff_length = 1

        if self.multi_steps > 1:
            step_rewards, step_states, step_actions = self._episode_rewards_states_actions(batch)
            rewards, states, actions, \
                batch_eff_length = self._multi_step_batch(step_rewards, step_states, step_actions)

        return states, actions, rewards, next_states, dones, batch_eff_length