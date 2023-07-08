from typing import Tuple
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
        else:
            # 处理梯子的值，梯子的值不能超过格子数，让梯子无重复反向赋值
            ladders =  set(np.random.randint(1, self.SIZE, size=ladder_num))
            while len(ladders) < self.ladder_num * 2:
                ladders.add(np.random.randint(1, self.SIZE))

            ladders = list(ladders)
            ladders = np.array(ladders)
            np.random.shuffle(ladders)
            ladders = ladders.reshape((self.ladder_num, 2))

            re_ladders = list()
            for i in ladders:
                re_ladders.append([i[1],i[0]])
            
            re_ladders = np.array(re_ladders)
            self.ladders = dict(np.append(re_ladders, ladders, axis=0))

        print(f"ladders: {self.ladders}, dice: {self.dices}")
        self.pos = 1
    
    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, a):
        step = np.random.randint(1, self.dices[a]+1)
        self.pos += step
        if self.pos == 100:
            return 100 , 100 , 1, {} # pos， reward， done， info
        elif self.pos > 100:
            self.pos = 200 - self.pos
        
        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        
        return self.pos, -1, 0, {} # pos， reward， done， info
    

    def reward(self, s):
        if s == 100 :
            return 100
        return -1
    
    def render(self):
        pass



def Table_Agent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

 
