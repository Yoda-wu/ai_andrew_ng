import numpy as np
import math
import torch
from collections import defaultdict

class QLearning(object):


    def __init__(self, n_actions, n_state, cfg):
        self.n_actions = n_actions # 动作空间
        self.n_state = n_state     # 状态空间
        self.lr = cfg.lr            # 学习率
        self.gamma = cfg.gamma      # 折扣因子
        self.samples_count = 0    # 采样次数
        self.epsilon_start = cfg.epsilon_start  # epsilon-greedy策略中的epsilon
        self.epsilon_end  = cfg.epsilon_end     # epsilon-greedy策略中的终止epsilon
        self.epsilon_decay = cfg.epsilon_decay # epsilon衰减率
        self.epsilon = self.epsilon_start # 当前epsilon
        print(n_actions,n_state)
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions)) # Q表 嵌套字典存放 状态-> 动作-> 状态-动作值的映射


    def samples(self, state):
        '''采样动作'''
        self.samples_count += 1
        # greedy衰减，因为训练初期agent选取动作不稳定，因此需要更为大胆的探索，后期训练趋于稳定，就可以加大对reward比较大动作采取
        self.epsilon = self.epsilon_end  + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.samples_count / self.epsilon_decay)

        # epsilon-greedy策略
        if np.random.uniform(0, 1) > self.epsilon:
            # argmax
            action = np.argmax(self.Q_table[str(state)])
        else:
            # random
            action = np.random.choice(self.n_actions) # 随机选择动作
        # print(f'action is {action}, n_actions is {self.n_actions}')
        return action
    
    def update(self, state, action, reward, next_state, terminated):
        Q_predict = self.Q_table[str(state)][action]
        if terminated:
            Q_predict = reward
        else :
            Q_predict = reward + self.gamma * np.max(self.Q_table[str(next_state)]) # 贪婪策略
        self.Q_table[str(state)][action] += self.lr * (Q_predict - self.Q_table[str(state)][action]) # 更新Q表

    def predict(self, state):
        return np.argmax(self.Q_table[str(state)])