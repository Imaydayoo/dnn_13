import math
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

from data_handler import DataHander
from initializer import Initializer
from net import DNNNet
from optimizer import Optimizer

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
if __name__ == "__main__":
    # fetal_weight_data = pd.read_csv("data/housing.data", header=0, index_col=None, sep='\s+')
    fetal_weight_data = pd.read_csv('/Users/apple/Desktop/深度学习数据/华侨医院-孕产妇EHR信息(脱敏)/data-13.csv')
    print("data_shape:", fetal_weight_data.shape)

    data_sample = fetal_weight_data.iloc[:, :-1].values
    data_label = fetal_weight_data.iloc[:, -1].values.reshape(-1, 1)

    mean = data_sample.mean(axis=0)
    std = data_sample.std(axis=0)
    data_sample = (data_sample - mean) / std

    data_length = data_label.shape[0]
    train_data_length = int(data_length * 0.8)
    print("train_label_length:", train_data_length)
    data_sample_train, data_sample_test = data_sample[:train_data_length], data_sample[train_data_length:]
    data_label_train, data_label_test = data_label[:train_data_length], data_label[train_data_length:]
    num_iterations = 100  # n_t = 1000 w_d = 0.01也有一样的效果
    lr = 0.15
    weight_decay = 0.0001
    train_batch_size = 500
    test_batch_size = 500  # 16
    data_handler = DataHander(500)  # 16
    opt = Optimizer(lr=lr, momentum=0.9, iteration=0, gamma=0.0005, power=0.75)
    initializer = Initializer()
    data_handler.get_data(sample=data_sample_train, label=data_label_train)
    data_handler.shuffle()
    dnn = DNNNet(optimizer=opt.batch_gradient_descent_anneling, initializer=initializer.xavier,
                 batch_size=train_batch_size, weights_decay=weight_decay)
    dnn.initial()
    train_error = []
    max_loss = math.inf
    early_stopping_iter = 15000
    early_stopping_mark = 0
    for i in range(num_iterations):
        print('第', i, '次迭代')
        opt.update_iteration(i)
        data_handler.pull_data()
        dnn.forward_train(data_handler.output_sample, data_handler.output_label)
        # 打印每次迭代的准确率

        dnn.backward_train()
        dnn.update()
        train_error.append(dnn.loss.loss)
        if max_loss > dnn.loss.loss:
            early_stopping_mark = 0
            max_loss = dnn.loss.loss
        if early_stopping_mark > early_stopping_iter:
            break
        early_stopping_mark += 1
    plt.plot(train_error)
    plt.show()
    # 测试
    dnn.eval(data_sample_test, data_label_test)
