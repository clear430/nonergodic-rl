import numpy as np
from typing import List, Tuple

class ReplayBuffer():
    """
    Experience replay buffer with uniform sampling based on 
    https://link.springer.com/content/pdf/10.1023%2FA%3A1022628806385.pdf.
    Multiplicative dyanmics and non-ergodic details are discussed in
    https://aip.scitation.org/doi/pdf/10.1063/1.4940236.

    Methods:
        store_exp(state, action, reward, next_state, done):
            Stores current step into experience replay buffer.

        _construct_history(step, epis_history):
            Generate reward and state history for a sampled step.

        _episode_rewards_states(batch):
            Collect reward and state histories for the batch.

        _multi_step_rewards_and_states(reward_history, state_history, multi_length):
            Generate multi-step rewards and intial state for a sampled step.

        _multi_step_batch(step_rewards, step_states):
            Collect multi-step rewards and intial states for the batch.
        
        get_sample_exp():
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

        self.dyna = str(inputs_dict['dynamics'])
        self.game_over = inputs_dict['game_over']
        self.initial_reward = 0 if self.dyna == 'A' else inputs_dict['initial_reward']
        self.unique_hist = str(inputs_dict['unique_hist'])
        self.compunding = str(inputs_dict['compounding'])
        self.r_abs_zero = inputs_dict['r_abs_zero']
        self.zero_scale = inputs_dict['random']

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
        self.alive_memory = np.zeros(self.mem_size, dtype=np.bool8)

        # required for intialisation
        self._contruct_history(1, 0)
        self._episode_rewards_states([])
        self._multi_step_rewards_and_states(np.zeros((1)), np.zeros((1,1)), 1)

    def store_exp(self, state: np.ndarray, action: np.ndarray, reward: float, 
                  next_state: np.ndarray, done: bool):
        """
        Store a transistion to the buffer containing a total up to a maximum size and log 
        history of rewards and states for each episode.

        Paramters:
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
        if self.multi_steps > 1 or self.dyna == 'M':
            self.epis_idx[-1] = idx

            # aggregate the reward and state histories of the currently trained episode
            try:
                current_start = self.epis_idx[-2] + 1
                current_reward_memory = self.reward_memory[current_start:idx + 1]
                current_state_memory = self.next_state_memory[current_start:idx + 1]

            except:
                try:
                    # used for the start of a new training episode (excluding the first)
                    current_reward_memory = self.reward_memory[0:idx + 1]
                    current_state_memory = self.next_state_memory[0:idx + 1]

                except:
                    # used for the the very first training step of the first episode
                    current_reward_memory = self.reward_memory[idx]
                    current_state_memory = self.next_state_memory[idx]
            
            # log the aggregated history upon termination of the training episode
            if done is True:
                self.epis_idx.append(idx + 1)
                self.epis_reward_memory.append(current_reward_memory)
                self.epis_state_memory.append(current_state_memory)

            if self.dyna == 'M':
                # generate log of when the cumulative episode reward exceeds the minimum threshold
                try:
                    alive = bool(1) if self.initial_reward + np.sum(self.reward_memory[current_start:idx + 1]) > self.game_over else bool(0)
                except:
                    alive = bool(1) if self.initial_reward + np.sum(self.reward_memory[0:idx + 1]) > self.game_over else bool(0) 
                
                self.alive_memory[idx] = alive

                # treat each episode as one complete history
                if self.unique_hist == 'N':
                    pass 
         
        self.mem_idx += 1

    def _contruct_history(self, step: int, epis_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Given a single mini-batch sample (or step), obtain the history of rewards, states,
        and whether previous states where 'alive' (non-ergodicity).

        Parameters:
            step: index or step of sample step
            epis_history: index list containing final steps of all training epsiodes

        Returns:
            rewards: sample reward history
            states: sample state history
            alive_states: sample life history
        """
        # find which episode the sample step is located
        try:
            sample_idx = int(np.max(np.where(step - epis_history > 0)) + 1) if step > epis_history[0] else 0
            n_rewards = step - epis_history[sample_idx - 1] if step > epis_history[0] else step
        except:
            # required for intialisation 
            sample_idx = 0
            n_rewards = 0

        # generate history of episode up till the sample step
        try:
            rewards = self.epis_reward_memory[sample_idx][0:n_rewards + 1]
            states = self.epis_state_memory[sample_idx][0:n_rewards + 1]
        except:
            try:
                # used for the first episode
                rewards = self.epis_reward_memory[0][0:n_rewards + 1]
                states = self.epis_state_memory[0][0:n_rewards + 1]               
            except:
                try:
                    # used for the the very first training step of the first episode
                    rewards = self.epis_reward_memory[0][0]
                    states = self.epis_state_memory[0][0]
                except:
                    # required for intialisation
                    rewards = 0
                    states = 0
                    
            
        if self.dyna == 'A':
            alive_states = False
            return rewards, states, alive_states

        else:
            # log of whether cumulative episode reward exceeds minimum threshold for step 
            try:
                alive_states = self.alive_memory[sample_idx:n_rewards + 1]
            except:
                try:
                    alive_states = self.alive_memory[0:n_rewards + 1]
                except:
                    try:
                        alive_states = self.alive_memory[0]
                    except:
                        alive_states = False

        return rewards, states, alive_states
    
    def _episode_rewards_states(self, batch: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect respective histories for each sample in the mini-batch.

        Parameters:
            batch: list of mini-batch steps representing samples
        
        Returns:
            sample_epis_rewards: mini-batch reward history
            sample_epis_states: mini-batch state history
            sample_epis_alive: mini-batch life history
        """
        epis_history = np.array(self.epis_idx)
        batch_histories = [self._contruct_history(step, epis_history) for step in batch]
        sample_epis_rewards = [x[0] for x in batch_histories]
        sample_epis_states = [x[1] for x in batch_histories]

        if self.dyna == 'A':
            sample_epis_alive = False
            return sample_epis_rewards, sample_epis_states, sample_epis_alive

        else:
            sample_epis_alive = [x[2] for x in batch_histories]

        return sample_epis_rewards, sample_epis_states, sample_epis_alive

    def _multi_step_rewards_and_states(self, reward_history: np.ndarray, state_history: np.ndarray, 
                                       multi_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For a single mini-batch sample, generate multi-step rewards and identify intial state.

        Parameters:
            reward_history: entire reward history of sample
            state_history: entire state history of sample
            multi_length: minimum of length of episode or multi-steps

        Returns:
            multi_reward: discounted sum of multi-step rewards
            intial_state: array of intial state before bootstrapping
            cum_rewards: undiscounted sum of rewards
        """
        idx = int(multi_length)

        # the sampled step is treated as the (n-1)th step  
        multi_reward = sum([self.gamma**t * reward_history[-idx + t] for t in range(idx - 1)])
        intial_state = state_history[-idx]
 
        if self.dyna == 'A':
            cum_reward = False
            return multi_reward, intial_state, cum_reward

        else:
            cum_reward = sum([reward_history[-idx + t] for t in range(idx - 1)])

        return multi_reward, intial_state, cum_reward
    
    def _multi_step_batch(self, step_rewards: List[np.ndarray], step_states: List[np.ndarray]) \
                          -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect respective multi-step returns and intial states for each sample in the mini-batch.

        Parameters:
            step_rewards: complete reward history of entire mini-batch
            step_states: complete state history of entire mini-batch

        Returns:
            batch_multi_reward: discounted sum of multi-step rewards
            batch_intial_state: array of intial state before bootstrapping
            batch_cum_rewards: undiscounted sum of rewards
        """
        # length taken to be the minimum of either history length or multi-steps
        length = np.minimum(np.array([x.shape[0] for x in step_rewards]), self.multi_steps)

        batch_multi = [self._multi_step_rewards_and_states(step_rewards[x], step_states[x], length[x]) 
                       for x in range(self.batch_size)]

        batch_multi_rewards = np.array([batch_multi[x][0] for x in range(self.batch_size)])
        batch_states = np.array([batch_multi[x][1] for x in range(self.batch_size)])
        
        if self.dyna == 'A':
            batch_cum_rewards = False
            return batch_multi_rewards, batch_states, batch_cum_rewards

        else:
            batch_cum_rewards = np.array([batch_multi[x][2] for x in range(self.batch_size)])

        return batch_multi_rewards, batch_states, batch_cum_rewards

    def sample_exp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniformly sample a batch from replay buffer for agent learning

        Returns:
            states: batch of environment states
            actions: batch of continuous actions taken to arrive at states
            rewards: batch of (discounted multi-step)  rewards from current states
            next_states: batch of next environment states
            dones (bool): batch of done flags
            epis_rewards: batch of cumulative rewards up till previous states
        """
        if self.dyna == 'A':
            # pool batch from either partial or fully populated buffer
            max_mem = min(self.mem_idx, self.mem_size)
            batch = np.random.choice(max_mem, size=self.batch_size)

            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            next_states = self.next_state_memory[batch]
            dones = self.terminal_memory[batch]
            epis_rewards = False

            if self.multi_steps > 1:
                step_rewards, step_states, _ = self._episode_rewards_states(batch)
                rewards, states, _ = self._multi_step_batch(step_rewards, step_states)

            return states, actions, rewards, next_states, dones, epis_rewards

        else:
            # pool batch from 'living' states
            alive_mem = np.where(self.alive_memory == True)[0]
            batch = np.random.choice(alive_mem, size=self.batch_size)
            # print(alive_mem)

            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            next_states = self.next_state_memory[batch]
            dones = self.terminal_memory[batch]

            step_rewards, step_states, step_alive = self._episode_rewards_states(batch)
            epis_rewards = np.array([self.initial_reward + np.sum(x) for x in step_rewards]) - rewards

            if self.multi_steps > 1:
                epis_rewards += rewards
                rewards, states, cum_rewards = self._multi_step_batch(step_rewards, step_states)
                epis_rewards -= cum_rewards
            
        return states, actions, rewards, next_states, dones, epis_rewards