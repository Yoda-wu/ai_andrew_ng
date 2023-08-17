import numpy as np
import matplotlib.pylab as plt

class BernoulliBandit:
    """伯努利多臂老虎机"""
    def __init__(self, K):
        self.K = K
        self.probs = np.random.uniform(size = K) # 随机生成K个概率，作为拉动每根手臂的获奖

        self.best_idx = np.argmax(self.probs) # 获奖概率最大的手臂
        self.best_prob = self.probs[self.best_idx] # 最大的的获奖概率
    
    def step(self, k):
        # 玩家选择k号手臂，根据该老虎机的k号手臂的获奖概率返回1或者0
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
    
class MAB_Solver:
    """
    多臂老虎机算法
        - 根据策略选择动作
        - 根据动作获取奖励
        - 更新期望奖励估值
        - 更新累积懊悔和计数
    """
    def __init__(self, bandit) -> None:
        self.bandit = bandit
        self.counts =  np.zeros(self.bandit.K) # 每根手臂被拉动的次数
        self.regret = 0 # 当前步的累积懊悔
        self.actions = [] # 每一步选择的动作
        self.regrets = [] # 每一步的累积懊悔

    def update_regret(self, k):
        # print(k)
        # 更新累积懊悔, k 为本次动作选择的手臂
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        # print(self.bandit.probs[k])
        # print(len(self.regret[0]))

        self.regrets.append(self.regret)
    
    def run_one_step(self):
        """根据策略选择动作、根据动作获取奖励和更新期望奖励估值"""
        raise NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] +=1
            self.actions.append(k)
            self.update_regret(k)
            


class EpsilonGreedyMAB(MAB_Solver):
    """ epsilon-greedy 多臂老虎机算法"""
    def __init__(self, bandit, epsilon=0.01, init_prob = 1.0) -> None:
        super(EpsilonGreedyMAB, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array(self.bandit.K * [init_prob]) # 期望奖励估值

    def run_one_step(self):
        
        if np.random.random() < self.epsilon:
            k = np.random.randint(0 , self.bandit.K)# 随机选择一个手臂
        else: 
            k = np.argmax(self.estimates) # 选择期望奖励估值最大的手臂
        
        r = self.bandit.step(k) # 根据选择的手臂获取奖励
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k]) # 更新期望奖励估值
        return k

class DecayEpsilonGreedyMAB(MAB_Solver):
    """epsilon衰减的epsilon-greedy多臂老虎机算法"""
    def __init__(self, bandit, init_prob=1.0) -> None:
        super(DecayEpsilonGreedyMAB, self).__init__(bandit)
        self.estimates = np.array(self.bandit.K * [init_prob]) # 期望奖励估值
        self.total_count = 0 

    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K) # 随机选择一个手臂

        else:
            k = np.argmax(self.estimates) # 选择期望奖励估值最大的手臂
        
        r = self.bandit.step(k) # 根据选择的手臂获取奖励
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k]) # 更新期望奖励估值
        return k 

class UCBMAB(MAB_Solver):
    """ UCB 多臂老虎机算法"""

    def __init__(self, bandit, coef, init_prob=1.0) -> None:
        super(UCBMAB,self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array(self.bandit.K * [init_prob]) # 期望奖励估值
        self.coef = coef # UCB算法的超参数 表示不确定性

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * self.counts + 1 )
        )# 计算每个手臂的UCB值
        k = np.argmax(ucb) # 选择UCB值最大的手臂
        r = self.bandit.step(k) # 根据选择的手臂获取奖励
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k]) # 更新期望奖励估值
        return k


class ThompsonSamplingMAB(MAB_Solver):
    def __init__(self, bandit):
        super(ThompsonSamplingMAB, self).__init__(bandit)
        self._a = np.ones(self.bandit.K) # 每根手臂奖励为1 的次数
        self._b = np.ones(self.bandit.K) # 每根手臂奖励为0 的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)
        self._a[k] += r
        self._b[k] += 1 - r

        return k




def plot_results(solvers, solve_names):
    """绘制多臂老虎机算法的累积懊悔"""
    
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        print(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solve_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
        (bandit_10_arm.best_idx, bandit_10_arm.best_prob))
    
    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [
        EpsilonGreedyMAB(bandit_10_arm, epsilon=e) for e in epsilons
    ]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)

    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
    
    
    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedyMAB(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayEpsilonGreedyMAB(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    np.random.seed(1)
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCBMAB(bandit_10_arm, coef)
    UCB_solver.run(5000)
    print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    plot_results([UCB_solver], ["UCB"])


    np.random.seed(1)
    thompson_sampling_solver = ThompsonSamplingMAB(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])