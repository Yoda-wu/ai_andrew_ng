import gym
import QLearning
import CliffWalkingWapper
import Config
cfg = Config.Config()
'''初始化环境'''  
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
env = CliffWalkingWapper(env)
agent = QLearning(
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,
    cfg=cfg
)
rewards = []  
ma_rewards = [] # moving average reward
for i_ep in range(cfg.train_eps): # train_eps: 训练的最大episodes数
    ep_reward = 0  # 记录每个episode的reward
    state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    while True:
        action = agent.samples(state)  # 根据算法选择一个动作
        next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
        agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
        state = next_state  # 存储上一个观察值
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
    else:
        ma_rewards.append(ep_reward)
    print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))
