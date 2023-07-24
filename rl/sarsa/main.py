from  sarsa_agent import SarsaAgent
from  sarsa_env import WindyGridWorld
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from  utils.Config import Config


def env_agent_config(cfg , isEight = False, isNinth = False  , seed = 1):
    env = WindyGridWorld(isEight, isNinth)
    n_states = env.max_x * env.max_y
    n_actions = env.action_spaces
    print(f'状态维度：{n_states}, 动作维度：{n_actions}')
    agent = SarsaAgent(n_states, n_actions, cfg)
    return env, agent


def game(cfg, isEight = False, isNinth = False ):
    env, agent = env_agent_config(cfg, isEight, isNinth)
    hist = []
    EPISODES = 200
    for episode in range(EPISODES):
        state = 1, 4
        action = agent.samples(state)
        for step in range(5000):
            next_state, reward, terminated, info = env.step(state, action)
            next_action = agent.samples(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)
            state = next_state
            action = next_action
            if reward == 0 :
                break
        state = 1, 4
        action = agent.predict(state)
        for step in range(5000):
            next_state, reward, terminated, info = env.step(state, action)
            next_action = agent.predict(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)
            state = next_state
            action = next_action
            if reward == 0 :
                hist.append(step)
                break
    return hist

def train(cfg, env, agent) :
     
    rewards = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励
        state = 1,4   # 重置环境,即开始新的回合
        while True:
            action = agent.samples(state)  # 根据算法采样一个动作
            # print(f"state:{state}, action:{action}")
            next_state, reward, terminated, info = env.step(action)  # 与环境进行一次动作交互
            agent.update(state, action, reward, next_state, terminated)  # Q学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated:
                break
        rewards.append(ep_reward)
        if (i_ep+1)%20==0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon:.3f}")
    print('完成训练！')
    return {"rewards":rewards}


if __name__ == '__main__':
    cfg = Config()
    hist_4 = []
    hist_8 = []
    hist_9 = []


    for i in range(200):
        print(f'第{i}次训练')
        hist_4.append(game(cfg, isEight=False, isNinth= False))
        hist_8.append(game(cfg, isEight = True, isNinth=False))
        hist_9.append(game(cfg, isEight = True, isNinth = True))
    
    hist_4 = np.mean(hist_4, axis = 0)
    hist_8 = np.mean(hist_8, axis = 0)
    hist_9 = np.mean(hist_9, axis = 0)


    plt.style.use('dark_background')
    plt.figure(figsize=(10,10))
    plt.title("4 vs 8 vs 9 action space in wind world")
    plt.xlabel('epsiodes', fontsize='xx-large')
    plt.ylabel('steps', fontsize='xx-large')
    plt.plot(hist_4, '-', c = 'r', label='4 actions')
    plt.plot(hist_8, '-', c = 'g', label='8 actions')
    plt.plot(hist_9, '-', c = 'b', label='9 actions')
    plt.legend(loc = 'best', prop = {'size': 12})
    plt.show()