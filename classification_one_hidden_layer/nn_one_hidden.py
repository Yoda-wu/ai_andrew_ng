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


def cost_function(A2, Y, parameters):
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
    w1 = parameters["w1"]
    w2 = parameters["w2"]

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

def backward(parameters, cache, X, Y):
    """
    input:
        parameters - 神经网络的参数
        cache - 包含Z1,A1,Z2,A2
        X - 输入数据
        Y - 标签
    output:
        grads - 梯度向量
    """
    m = X.shape[1]

    w1 = parameters["w1"]
    w2 = parameters["w2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A - Y
    dW2 = (1 / m ) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(w2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1,keepdims=True)
    grads = {
        "dW1" : dW1,
        "db1" : db1,
        "dW2" : dW2,
        "db2" : db2
        
    }
    return grads

def update_prarm(parameters, grads, rate=0.5):
    """
    input:
        - parameters 参数
        - grads 反向传播梯度
        - rate 学习率
    output
        - parameters 更新之后的参数
    """
    w1, w2 = parameters["w1"], parameters["w2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    w1 = w1 - rate * dW1
    b1 = b1 - rate * db1
    w2 = w2 - rate * dW2
    b2 = b2 - rate * db2

    parameters = {
        "w1" :w1,
        "b1" :b1,
        "w2" :w2,
        "b2" :b2
    }
    return parameters



def nn_model(X, Y, n_h, iterations, print_cost = False):
    """
    input:
        - X 数据集
        - Y 标签
        - n_h 隐藏层节点数
        - iterations 迭代次数
        - print_cost 打印成本数值
    output:
        - paramters 模型学习参数
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = init_param(n_x, n_h, n_y)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    for i in range(iterations):
        A2, cache = forward(X, parameters)
        cost = cost_function(A2, Y, parameters)
        grads = backward(parameters, cache, X, Y)
        parameters = update_prarm(parameters, grads, rate=0.5)

        if print_cost and i % 1000 == 0:
            print(f"第{i}次迭代, cost={cost }")
    return parameters


def predict(parameters, X):
    """
    input:
        - paramters 模型学习参数
        - X 输入数据
    output:
        - predictions - 模型预测向量
    """
    A2, cache = forward(X, parameters)
    predictions = np.round(A2)
    return predictions
parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

#绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')


plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] #隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
