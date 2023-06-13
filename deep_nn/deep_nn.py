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

def L_model_forward(X, parameters):
    """
    实现[LINEAR->RELU]*(L-1)->LINEAR->SIGMOID前向传播
    input:
        X - 数据
        parameters - init_param_deep()的输出
    output:
        AL - 最后的激活值
        caches - 包含每一层的cache的列表
    """
    caches = []
    A = X
    L = len(parameters) // 2 # 网络层数
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['w' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    """
    计算成本
    input:
        AL - 预测值 维度为(1, 实例数量)
        Y - 标签
    output:
        cost - 成本
    """
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y)) / m
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    """
    实现线性部分的反向传播
    input:
        dZ - 相对于（当前第l层的）线性输出的成本梯度
        cache - 来自当前层前向传播的值的元组(A_prev, W, b)
    output:
        dA_prev - 相对于激活(前一层l-1)的成本梯度，与A_prev维度相同
        dW - 相对于W(当前层l)的成本梯度，与W维度相同
        db - 相对于b(当前层l)的成本梯度，与b维度相同
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation="relu"):
    """
    实现LINEAR->ACTIVATION层的反向传播
    input:
        dA - 当前层l的激活后的梯度值
        cache - 来自当前层前向传播的值的元组(linear_cache, activation_cache)
        activation - 激活函数类型 string("sigmoid" or "relu")
    output:
        dA_prev - 相对于激活(前一层l-1)的成本梯度，与A_prev维度相同
        dW - 相对于W(当前层l)的成本梯度，与W维度相同
        db - 相对于b(当前层l)的成本梯度，与b维度相同
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    对[LINEAR->RELU]*(L-1)->LINEAR->SIGMOID反向传播
    input:
        AL - 概率向量，正向传播的输出
        Y - 标签向量
        caches - 包含每一层cache的列表
    返回:
        grads - 包含每一层梯度值的字典
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    current_cache = caches[L-1]
    grads['dA' + str(L-1)], grads['dw' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dw_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, activation="relu")
        grads['dA' + str(l)] = dA_prev_temp
        grads['dw' + str(l+1)] = dw_temp
        grads['db' + str(l+1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数
    input: 
        parameters - 包含参数的字典
        grads - 包含梯度值的字典
        learning_rate - 学习率
    output:
        parameters - 包含更新参数的字典
    """
    L = len(parameters)
    for l in range(L):
        parameters['w' + str(l+1)] = parameters['w' + str(l+1)] - learning_rate * grads['dw' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]
    return parameters

def two_layer_model(X,Y,layers_dims, rate=0.0075, iteration=3000, print_cost=False, isPlot=True):
    
    """
    实现一个两层的神经网络：LINEAR->RELU->LINEAR->SIGMOID
    input:
        X - 输入数据，维度为(n_x, 数量)
        Y - 标签，维度为(1, 数量)
        layers_dims - 层数的列表，维度为(n_x, n_h, n_y)
        rate - 学习率
        iteration - 迭代次数
        print_cost - 是否打印成本值
        isPlot - 是否绘制出误差值的图谱
    output:
        parameters - 训练好的参数
    """
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x, n_h, n_y) = layers_dims

    parameters = init_param(n_x, n_h, n_y)

    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    for i in range(0, iteration):
        A1, cache1 = linear_activation_forward(X, w1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, w2, b2, activation="sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = -(np.divide(Y, A2) - np.divide(1-Y, 1-A2))

        dA1, dw2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dw1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

        grads['dw1'] = dw1
        grads['dw2'] = dw2
        grads['db1'] = db1
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, rate)
        w1 = parameters['w1']
        w2 = parameters['w2']
        b1 = parameters['b1']
        b2 = parameters['b2']

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i ,"次迭代，成本值为：", np.squeeze(cost))
            
    if isPlot:
       plt.plot(np.squeeze(costs))
       plt.ylabel('cost')
       plt.xlabel('iterations (per tens)')
       plt.title("Learning rate =" + str(rate))
       plt.show()

    return  parameters


def predict(X, y, paramters):
    """
    预测L层神经网络的结果
    input:
        X - 测试集
        y - 标签
        paramters - 训练好的参数
    output:
        p - 给定数据集X的预测
    
    """

    m = X.shape[1]
    n = len(paramters) // 2
    p = np.zeros((1, m))

    probs, caches = L_model_forward(X, paramters)
    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为：" + str(float(np.sum((p == y)) / m)))
    return p


def L_layer_model(X, Y, layers_dims, rate = 0.0075, iterations=3000, print_cost= False, isPlot=True):
    """
    实现一个L层神经网络：[LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    input:
        X - 输入数据，维度为(n_x, 数量)
        Y - 标签，维度为(1, 数量)
        layers_dims - 层数的列表，维度为(n_x, n_h, n_y)
        rate - 学习率
        iteration - 迭代次数
        print_cost - 是否打印成本值
        isPlot - 是否绘制出误差值的图谱
    output:
        parameters - 训练好的参数
    
    """

    np.random.seed(1)
    costs = []
    parameters = init_param_deep(layers_dims)
    
    for i in range(0, iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i ,"次迭代，成本值为：", np.squeeze(cost))
            
    if isPlot:
         plt.plot(np.squeeze(costs))
         plt.ylabel('cost')
         plt.xlabel('iterations (per tens)')
         plt.title("Learning rate =" + str(rate))
         plt.show()
    return parameters


def print_mislabeled_images(classes, X, y, p):
    """
	绘制预测和实际不同的图像。
	    X - 数据集
	    y - 实际的标签
	    p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


print_mislabeled_images(classes, test_x, test_y, pred_test)


