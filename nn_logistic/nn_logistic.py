# 搭建具有神经网络思维的Logistic回归
# 我们要做的事是搭建一个能够**【识别猫】** 的简单的神经网络
import numpy as np
import h5py
import matplotlib.pyplot as plt
from  lr_utils import load_dataset
import warnings
warnings.filterwarnings("ignore")

def data_init():
    """
    加载并预处理数据集
    output:
        train_set_x - 训练集
        test_set_x  - 测试集
        train_set_y - 训练集标签
        test_set_y  - 测试集标签
        classes     - 分类
    """
    # 加载数据集
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # 查看数据的图片是什么情况
    index = 25
    # plt.imshow(train_set_x_orig[index])
    # plt.show()
    # 打印训练集的标签
    print("训练集的标签{}".format(train_set_y))

    # np.squeeze()是压缩维度的意思，即去掉维度为1的维度。
    # train_set_y[:, index] 是[1] 压缩之后是1
    print("index {}'s y = {}, it is a {}".format(index, train_set_y[:, index], classes[np.squeeze(train_set_y[:, index])].decode("utf-8")))
    # print("训练集的图片的维度{}".format(test_set_x_orig.shape)) train_set_x_orig 是一个维度为(m_​​train，num_px，num_px，3）
    m_train = train_set_y.shape[1] # 训练集里图片的数量。209
    m_test = test_set_y.shape[1] # 测试集里图片的数量。 50
    num_px = train_set_x_orig.shape[1] # 训练、测试集里面的图片的宽度和高度（均为64x64）。


    print (f"训练集的数量: m_train = {m_train}" )
    print (f"测试集的数量 : m_test ={m_test}")
    print (f"每张图片的宽/高 : num_px = {num_px}" )
    print (f"每张图片的大小 : ( {num_px} , {num_px} , 3 ) ")
    print (f"训练集_图片的维数 : {train_set_x_orig.shape}")
    print (f"训练集_标签的维数 : {train_set_y.shape}" )
    print (f"测试集_图片的维数: {test_set_x_orig.shape}")
    print (f"测试集_标签的维数: {test_set_y.shape}"  )

    # 需要将原始数据维度为（64,64,3)的矩阵降低维度，平展成(64*64*3,1)的数组
    # 当你想将形状（a，b，c，d）的矩阵X平铺成形状（b * c * d，a）的矩阵X_flatten时，可以使用以下代码
    train_set_x_flattern = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T # -1表示自动计算列数。 12288 = 64 * 64 * 3
    print("训练集降维之后的图片的维度{}".format(train_set_x_flattern.shape)) 
    test_set_x_flattern = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    print("测试集降维之后的图片的维度{}".format(test_set_x_flattern.shape))

    # 归一化
    # 由于图像数据是RGB值，那么最大值不超过255
    # 我们可以直接除255，将数据归一化到0-1之间
    train_set_x = train_set_x_flattern / 255
    test_set_x = test_set_x_flattern / 255
    return (train_set_x, test_set_x, train_set_y, test_set_y, classes)


# 构建网络

# sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 初始化参数
def init_param(dim):
    """
    z = w^T X + b
    创建一个维度为(dim,1)的w向量, b初始化为0
    返回
    w
    b
    """
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))
    return (w, b)

# 前向传播和反向传播
def propagate(w, b, X, Y):
    """
    首先前向传播和反向传的cost和梯度
    input:
        w - 权重参数 维度为(num_px * num_px * 3, 1)
        b - 偏置参数 
        X - 矩阵类型为(num_px*num_px*3, m)的数据集
        Y - 标签向量， 维度为(1, m)
    output:
        cost - 逻辑回归的负对数似然成本loos函数
        dw - w的梯度
        db - b的梯度
        
    """
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T, X) + b) # 计算激活值
    cost = ( - 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) # 计算损失函数
    dw = (1 / m) * np.dot(X, (A - Y).T) # 计算w的梯度
    db = (1 / m) * np.sum(A - Y) # 计算b的梯度

    assert(dw.shape == w.shape)
    # assert(db.shape == b.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost) # 压缩维度
    grads = {
        "dw":dw,
        "db":db
    }
    return grads, cost

def optimize(w, b, X, Y, iteration, rate, print_cost = False):
    """
    通过梯度下降来优化参数
    input:
        w - 权重参数 维度为(num_px * num_px * 3, 1)
        b - 偏置参数 
        X - 矩阵类型为(num_px*num_px*3, m)的数据集
        Y - 标签向量， 维度为(1, m)
        iteration - 迭代次数
        rate - 学习率
        print_cost - 是否每100次打印损失值
    output:
        params - 权重参数和偏置参数的字典
        grads - 梯度值
        costs -  优化期间的损失变化
    """
    costs = []
    for i in range(iteration):
        grad, cost = propagate(w, b, X, Y)

        dw = grad["dw"]
        db = grad["db"]

        w = w - rate * dw
        b = b - rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"迭代次数为{i}, 损失值为{cost}")
        
    params = {
        "w":w,
        "b":b
    }
    grads = {
        "dw":dw,
        "db":db
    }
    return (params, grads, costs)


def predict(w,b, X, thredhold = 0.5):
    """
    根据训练后的参数，预测测试集的结果
    input:
        w - 权重参数 维度为(num_px * num_px * 3, 1)
        b - 偏置参数 
        X - 矩阵类型为(num_px*num_px*3, m)的数据集
    output:
        Y_pred - 预测的标签向量
    """
    m = X.shape[1]
    Y_pred = np.zeros((1, m))  
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_pred[0, i] = 1 if A[0, i] >thredhold else 0
    assert(Y_pred.shape == (1, m))
    return Y_pred

def model(X_train, X_test, Y_train, Y_test, iteration=2000, rate = 0.5, threadhold = 0.5, print_cost = False):
    """
    构建逻辑回归模型
    input:
        X_train - 训练集
        X_test - 测试集
        Y_train - 训练集标签
        Y_test - 测试集标签
        iteration - 迭代次数
        rate - 学习率
        threadhold - 阈值
        print_cost - 是否打印损失值
    output:
        d - 模型相关信息的字典
    """

    w, b = init_param(X_train.shape[0])
    # 进行梯度下降训练
    param, grads, costs = optimize(w, b, X_train, Y_train, iteration, rate, print_cost)
    w, b = param["w"], param["b"]
    # 进行预测
    Y_pred_test = predict(w, b, X_test, threadhold)
    Y_pred_train = predict(w, b, X_train, threadhold)

    # 打印准确度
    print(f"训练集准确度为: {100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100}%")
    print(f"测试集准确度为: {100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100}%")

    d = {
        "costs":costs,
        "Y_pred_test":Y_pred_test,
        "Y_pred_train":Y_pred_train,
        "w":w,
        "b":b,
        "rate":rate,
        "iteration":iteration
    }
    return d


def plot_result(d) :
    """
    绘制损失函数图像
    input:
        d - 模型相关信息的字典(costs, Y_pred_test, Y_pred_train, w, b, rate, iteration)
    output:
        None - 绘制损失函数图像
    """
    costs = np.squeeze(d["costs"])
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iteration(per hundreds)")
    plt.title(f"Learning rate = {d['rate']}")
    plt.show()

if __name__  == "__main__":
    X_train, X_test, Y_train, Y_test, classes = data_init()
    d = model(X_train, X_test, Y_train, Y_test, iteration=2000, rate = 0.005, threadhold = 0.5, print_cost = True)
    plot_result(d)


