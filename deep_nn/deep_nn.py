import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils
import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)

# 初始化参数
def init_param(n_x, n_h, n_y):
    """
    初始两层网络参数
    input:
        n_x - 输入层节点数
        n_h - 隐藏层节点数
        n_y - 输出层节点数
    output:
        parameters - 包含参数的字典
            w1 - 隐藏层权重矩阵维度为(n_h, n_x)
            b1 - 偏置向量维度为(n_h, 1)
            w2 - 输出层权重矩阵维度为(n_y, n_h)
            b2 - 偏置向量维度为(n_y, 1)
    """

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h , 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y , 1))

    # 使用断言确保我的数据格式是正确的
    assert(w1.shape == (n_h , n_x))
    assert(b1.shape == (n_h , 1))
    assert(w2.shape == (n_y , n_h))
    assert(b2.shape == (n_y , 1))

    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }

    return parameters

def init_param_deep(layer_dims):
    """
    初始化多层网络参数
    input:
        layer_dims - 包含每层节点数的列表
    output:
        parameters - 包含参数为w1,b1,...,wL,bL的字典
            w1 - 权重矩阵维度为(layer_dims[l], layer_dims[l-1])
            b1 - 偏置向量维度为(layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # 网络层数

    for l in range(1, L):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        # 使用断言确保我的数据格式是正确的
        assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """
    实现前向传播线性部分
    input:
        A - 上一层的激活值
        W - 权重矩阵
        b - 偏置向量
    output:
        Z - 激活函数的输入
        cache - 一个包含"A", "W" and "b"的字典
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    实现前向传播的LINEAR->ACTIVATION
    input:
        A_prev - 上一层的激活值
        W - 权重矩阵
        b - 偏置向量
        activation - 激活函数类型 string("sigmoid" or "relu")
    output:
        A - 激活函数的输出
        cache - 一个包含"linear_cache" and "activation_cache"的字典
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache
