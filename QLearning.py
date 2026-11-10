import copy
import numpy as np

class QTable():
    def __init__(self, config):
        self.actions = ['up', 'right', 'down', 'left']        
        self.q_table = np.zeros((config.MAZE_SIZE*config.MAZE_SIZE, len(self.actions)))
        self.MAZE_SIZE = config.MAZE_SIZE
        self.maze = config.maze_setting
        self.End_point = config.End_point
        self.Start_pint = config.Start_point
        self.maze_setting = config.maze_setting
        self.discount_factor = config.discount_factor
        self.epsilon = config.epsilon

        self.learning_rate = getattr(config, 'q_learning_rate', 0.1) # alpha

        self.robot_pos = self.Start_pint

    def choose_action(self, state):
        # Epsilon-greedy: explore with probability epsilon, otherwise exploit.
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.actions))
        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == q_values.max())
        return np.random.choice(best_actions) # random tie-breaking

    def step(self, pos):
        old_pos = copy.copy(pos)
        state = pos[0] * self.MAZE_SIZE + pos[1]
        done_flags = False

        # < - TASK I: Action Selection Mechanism - >
        action = self.choose_action(state)
        # < - TASK I: Action Selection Mechanism - >

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
    
    def update_q_table(self, pos, action, new_pos, reward, done_flags):
        state = pos[0] * self.MAZE_SIZE + pos[1]
        next_state = new_pos[0] * self.MAZE_SIZE + new_pos[1]

        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])

        # < - TASK II: Qtable Updates Algorithm Design - >
        if done_flags: # goal reached: no future reward after a terminal state
            target = reward
        self.q_table[state, action] = predict + self.learning_rate * (target - predict)
        # < - TASK II: Qtable Updates Algorithm Design - >

    def epsilon_decline(self):
        self.epsilon = max(0.01, 0.995 * self.epsilon)