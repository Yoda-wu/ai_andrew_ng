import numpy as np
import gym
from gym.spaces import Discrete, Box
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} cost time: {end - start}")
    pass


class Snake_Env(gym.Env):
    SIZE = 100 # 100 格
    def __init__(self, ladder_num , dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.observation_space = Discrete(self.SIZE+1) # 观察空间
        self.action_space = Discrete(len(dices)) # 动作空间 —— 有多少种action

        if ladder_num == 0:
            self.ladders = {0 : 0}
        