# 建立一个带有一个隐含层的nn
# 和上一次logisticsde nn模型不同，上次一次只有输入和输出层

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1)


# 加载数据集
X, Y = load_planar_dataset() # X 是一个numpy矩阵， Y是一个numpy向量，对应着X的标签
# 绘制数据集
# plt.cm.Spectral 颜色映射是基于Y来决定
# plt.scatter(X[0, :], X[1,:], c = np.squeeze(Y), s = 40 , cmap=plt.cm.Spectral)  plt.cm.Spectral
# plt.show()


shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1] # 训练集数量

print(f"X的维度为{shape_X}")
print(f"Y的维度为{shape_Y}")
print(f"数据集里一共有{m}个数据")

# 查看简单的Logistics回归分类的效果
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X.T, Y.T)

# 绘制逻辑回归分类器的效果
# plot_decision_boundary(lambda x:clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# plt.show()
LR_predictions = clf.predict(X.T)
print(f"逻辑回归的准确性： {float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1- LR_predictions)) / float(Y.size) *100)} 正确标记的数据点所占的百分比")


# 搭建神经网络

def layer_sizes(X, Y):
    """
    input:
        X - 输入数据集, 维度为(输入的数量， 训练/测试的数量)
        Y - 标签，维度为(输出数量， 训练/测试数量)

    output:
        n_x - 输入层节点数
        n_h - 隐藏层节点数
        n_y - 输出层节点数
    """

    n_x = X.shape[0] # 输入层节点数
    n_h = 4 # 隐藏层节点数
    n_y = Y.shape[0] # 输出层的数量
    return (n_x, n_h, n_y)


def test_layer_size():
    print("=========================测试layer_sizes=========================")
    X_test, Y_test = layer_sizes_test_case()
    n_x, n_h ,n_y = layer_sizes(X_test, Y_test)
    print("输入层的节点数量为: n_x = " + str(n_x))
    print("隐藏层的节点数量为: n_h = " + str(n_h))
    print("输出层的节点数量为: n_y = " + str(n_y))

def init_param(n_x, n_h ,n_y):
    """
    input:
        n_x - 输入层节点数
        n_h - 隐藏层节点数
        n_y - 输出层节点数
    output:
        parameter - 神经网络参数的字典
            w1 - 隐藏层权重矩阵维度为(n_h, n_x)
            b1 - 偏置向量维度为(n_h, 1)
            w2 - 输出层权重矩阵维度为(n_y, n_h)
            b2 - 偏置向量维度为(n_y, 1)
    """
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    #使用断言确保我的数据格式是正确的
    assert(w1.shape == ( n_h , n_x ))
    assert(b1.shape == ( n_h , 1 ))
    assert(w2.shape == ( n_y , n_h ))
    assert(b2.shape == ( n_y , 1 ))
    
    paramters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }

    return paramters


def test_init_param():
    print("=========================测试initialize_parameters=========================")    
    n_x , n_h , n_y = initialize_parameters_test_case()
    parameters = initialize_parameters(n_x , n_h , n_y)
    print("w1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("w2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# 前向传播
def forward(X, paramters):
    """
    input:
        X - 维度为(n_x, m)的输入数据
        paramters - 神经网络的初始化参数
    output:
        A2 - 输出层的激活值
        cache - 包含Z1, A1, Z2, A2的字典 
    """ 
    w1 = paramters["w1"]
    b1 = paramters["b1"]
    w2 = paramters["w2"]
    b2 = paramters["b2"]

    # 前向传播
    Z1 = np.dot(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)
    assert(A2.shape == (1, X.shape[1]))
    cache = {
        "Z1" :Z1,
        "A1" :A1,
        "Z2" :Z2,
        "A2" :A2
    }
    return A2, cache

def test_forward():
    #测试forward_propagation
    print("=========================测试forward_propagation=========================") 
    X_assess, parameters = forward_propagation_test_case()
    A2, cache = forward(X_assess, parameters)
    print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))


def cost_function(A2, Y, paramters):
    """
    交叉熵成本
    input:
        A2 - 输出层的激活值
        Y  - 标签向量
        paramters - 神经网络的参数
    output:
        cost - 交叉熵成本的输出方程

    """
    m = Y.shape[1]
    w1 = paramters["w1"]
    w2 = paramters["w2"]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply( (1-Y), np.log(1-A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    assert(isinstance(cost, float))
    return cost

def test_cost_function():
    #测试compute_cost
    print("=========================测试compute_cost=========================") 
    A2 , Y_assess , parameters = compute_cost_test_case()
    print("cost = " + str(cost_function(A2,Y_assess,parameters)))



# 反向传播

