
from collections import defaultdict
import numpy as np


class SarsaAgent(object):


    def __init__(self, n_states, n_actions, cfg):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = cfg.lr
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.epsilon = self.epsilon_start
        self.gamma = cfg.gamma
        self.sample_count = 0
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))

    
    def samples(self, state):
        self.sample_count += 1
        # epsilon decay
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.epsilon_decay * self.sample_count)
        if np.random.uniform(0, 1)> self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else :
            action = np.random.choice(self.n_actions)
        return action


    def update(self, state, action, reward, next_state, next_action, terminated):
        next_Q = self.Q_table[str(state)][action]
        if reward == 0:
            next_Q = 0
        else:
            next_Q = self.Q_table[str(next_state)][next_action]
        self.Q_table[str(state)][action] += self.lr * (reward + self.gamma * next_Q - self.Q_table[str(state)][action])
        
    

    def predict(self, state):
        return np.argmax(self.Q_table[str(state)])
