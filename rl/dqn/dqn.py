import random
import numpy as np
import gym
import collections
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.rl_utils as rl_utils



# 首先定义经验回放池的类，主要包括添加数据和采样数据
class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 队列，先进先出

    def add(self, state, action, reward, next_state, done): 
        # 添加数据到经验回放池
        self.buffer.append((state,action, reward, next_state, done))

    def samples(self, batch_size):
        # 从经验回放池中采样batch_size个数
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        
