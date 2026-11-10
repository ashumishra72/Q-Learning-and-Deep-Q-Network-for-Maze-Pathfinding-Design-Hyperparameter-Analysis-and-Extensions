import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epsilon = config.epsilon
        self.MAZE_SIZE = config.MAZE_SIZE
        self.discount_factor = config.discount_factor
        self.actions = ['up', 'right', 'down', 'left'] 
        
        self.maze = config.maze_setting
        self.End_point = config.End_point
        self.Start_point = config.Start_point

        self.robot_pos = self.Start_point
        self.replay_buffer = ReplayMemory(capacity=config.capacity, config=config)
        
        # < - TASK III: DQN Architecture Design - >
        n_states = self.MAZE_SIZE * self.MAZE_SIZE # one-hot state input
        hidden_size = getattr(config, 'hidden_size', 64)
        self.network = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(self.actions))
        )
        # < - TASK III: DQN Architecture Design - >

        # Bug fix: copy.copy() shared the weight tensors with self.network, so the
        # "target" network always equalled the online network. deepcopy() makes an
        # independent copy, and its parameters are frozen (no gradients).
        self.target_net = copy.deepcopy(self.network)
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self.target_net.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def forward(self, pos):
        state = torch.eye(self.MAZE_SIZE * self.MAZE_SIZE)[pos[0] * self.MAZE_SIZE + pos[1]]
        return self.network(state)

    def run_state(self, state):
        return self.network(state)

    def evaluate_target_net(self, state):
        return self.target_net(state)

    def step(self, pos):
        old_pos = copy.copy(pos)
        done_flags = False
        if np.random.random() < self.epsilon:
            action = np.random.choice(len(self.actions))
        else:
            action = np.argmax(self.forward(pos).detach().numpy())
        
        if self.actions[action] == 'up':
            pos = [old_pos[0], max(old_pos[1] - 1, 0)]
        elif self.actions[action] == 'down':
            pos = [old_pos[0], min(old_pos[1] + 1, self.MAZE_SIZE - 1)]
        elif self.actions[action] == 'left':
            pos = [max(old_pos[0] - 1, 0), old_pos[1]]
        elif self.actions[action] == 'right':
            pos = [min(old_pos[0] + 1, self.MAZE_SIZE - 1), old_pos[1]]
        else:
            raise ValueError("Invalid action")

        if pos == self.End_point:
            reward = 10
            done_flags = True
        elif(self.maze[pos[0], pos[1]] == 1):
            reward = -10
        else:
            reward = -1

        self.robot_pos = pos if self.maze[pos[0], pos[1]] == 0 else self.robot_pos

        return old_pos, action, self.robot_pos, reward, done_flags

    def learn(self):
        states_mem, actions_mem, reward_mem, next_states_mem, done_flags = self.replay_buffer.sample(self.config.batch_size)
        current_q_values = self.run_state(states_mem).gather(1, actions_mem.unsqueeze(1)).squeeze(1)
        
        next_q_values = self.evaluate_target_net(next_states_mem).max(1)[0]
        
        # < - TASK IV: Expected Q-Value Calculation - >
        expected_q_values = (reward_mem + self.discount_factor * next_q_values * (1 - done_flags)).detach()
        # < - TASK IV: Expected Q-Value Calculation - >
        
        loss = self.config.loss_fn(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def target_update(self):
        self.target_net.load_state_dict(self.network.state_dict())

    def epsilon_decline(self):
        self.epsilon = max(0.01, 0.995 * self.epsilon)

class ReplayMemory(object):
    def __init__(self, capacity, config):
        self.MAZE_SIZE = config.MAZE_SIZE
        self.buffer = deque(maxlen=capacity)

    def push(self, pos, action, next_pos, reward, done_flags):
        state = np.eye(self.MAZE_SIZE * self.MAZE_SIZE)[pos[0] * self.MAZE_SIZE + pos[1]]
        next_state = np.eye(self.MAZE_SIZE * self.MAZE_SIZE)[next_pos[0] * self.MAZE_SIZE + next_pos[1]]
        self.buffer.append((state, action, reward, next_state, done_flags))

    def sample(self, batch_size):
        state, action, reward, next_state, done_flags = zip(*random.sample(self.buffer, batch_size))
        return torch.FloatTensor(np.array(state)), torch.LongTensor(action), torch.FloatTensor(reward), torch.FloatTensor(np.array(next_state)), torch.FloatTensor(done_flags)

    def __len__(self):
        return len(self.buffer)
