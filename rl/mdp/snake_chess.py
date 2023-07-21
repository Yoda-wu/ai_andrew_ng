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



class TableAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n
        
        self.r = [env.reward(s) for s in range(0, self.s_len)]
        # 确定性策略
        self.pi = np.zeros(self.s_len, dtype=int)
        # A x S x S
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=float)
        
        # 函数参数向量化，参数可以传入列表
        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)
        
        # based-model 初始化表格所有位置的概率p[A,S,S]
        for i, dice in enumerate(env.dices):
            prob = 1.0 / dice
            for src in range(1, 100):
                # 因为arange只给一个数字的时候，是从0开始取到end-1，所以在此处+1
                step = np.arange(dice) + 1
                step += src
                step = np.piecewise(step, [step>100, step<=100], [lambda x: 200-x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    # 在当前位置pos=src的情况下，采取i投掷色子的方式，得到最终位置dst
                    # 概率直接求和的方式是否合理？
                    self.p[i, src, dst] += prob
        
        # 因为src最多到99，所以p[:, 100, 100]是0，此处进行填补
        self.p[:, 100, 100] = 1
        self.value_pi = np.zeros((self.s_len))
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8
        
        
    def play(self, state):
        return self.pi[state]