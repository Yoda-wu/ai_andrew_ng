import torch


class Config :

    def __init__(self):
        self.env_name = 'CliffWalking-v0' # 环境名
        self.algo_name = 'Q-Learning' # 算法名
        self.train_eps = 400 # 训练回合数
        self.test_eps = 10 # 测试回合数
        self.max_steps = 20 # 每回合最大步数
        self.epsilon_start = 0.9 # e-greedy 策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy 策略中的终止epsilon
        self.epsilon_decay = 200 # e-greedy 策略中epsilon的衰减率
        self.gamma = 0.9 # reward的衰减率
        self.lr = 0.1 # learning rate
        self.seed = 42
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    

