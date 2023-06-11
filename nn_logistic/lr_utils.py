import numpy as np
import h5py
    
    
def load_dataset():
    """
    
    加载数据集
    return:
    train_set_x_orig:保存的是训练集里面的图像数据(本训练集有209张64x64的图像)。
    train_set_y_orig:保存的是训练集的图像对应的分类值(【0 | 1】,0表示不是猫,1表示是猫)。
    test_set_x_orig: 保存的是测试集里面的图像数据(本训练集有50张64x64的图像)。
    test_set_y_orig: 保存的是测试集的图像对应的分类值(【0 | 1】,0表示不是猫,1表示是猫)
    classes: 保存的是以bytes类型保存的两个字符串数据 数据为：[b'non-cat' b'cat']
    """
    train_dataset = h5py.File('D:\\school\\github\\ai_andrew_ng\\nn_logistic\datasets\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('D:\\school\\github\\ai_andrew_ng\\nn_logistic\datasets\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes