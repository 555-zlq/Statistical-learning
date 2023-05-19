import math
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Softmax(object):

    def __init__(self):
        self.learning_step = 0.000001           # 学习速率
        self.max_iteration = 100000             # 最大迭代次数
        self.weight_lambda = 0.01               # 衰退权重


    def cal_e(self,x,l):

        theta_l = self.w[l]                     # 取出第l行，即第l个分类器的权重
        product = np.dot(theta_l,x)            # 计算theta_l和x的点积

        return math.exp(product)              # 返回e的指数

    def cal_probability(self,x,j):            # 计算第j个分类器对x分类的概率

        molecule = self.cal_e(x,j)           # 分子, e的指数
        denominator = sum([self.cal_e(x,i) for i in range(self.k)]) # 分母

        return molecule/denominator        # 返回概率


    def cal_partial_derivative(self,x,y,j):

        first = int(y==j)                           # 计算示性函数
        second = self.cal_probability(x,j)          # 计算后面那个概率

        return -x*(first-second) + self.weight_lambda*self.w[j]

    # 预测标签, x是测试集的一个实例
    def predict_(self, x):
        result = np.dot(self.w,x)
        row, column = result.shape

        # 找最大值所在的列
        _positon = np.argmax(result)
        m, n = divmod(_positon, column)

        return m

    def train(self, features, labels):
        self.k = len(set(labels))

        self.w = np.zeros((self.k,len(features[0])+1))
        time = 0

        while time < self.max_iteration:
            print('loop %d' % time)
            time += 1
            index = random.randint(0, len(labels) - 1)

            x = features[index]
            y = labels[index]

            x = list(x)
            x.append(1.0)
            x = np.array(x)

            derivatives = [self.cal_partial_derivative(x,y,j) for j in range(self.k)]

            for j in range(self.k):
                self.w[j] -= self.learning_step * derivatives[j]

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)

            x = np.matrix(x)
            x = np.transpose(x)

            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    print('Start read data')

    time_1 = time.time()

    # raw_data = pd.read_csv('/home/carton/workspace/python/Statistical-learning/database/homework4/train.csv', header=0)
    raw_data = pd.read_csv('../../database/homework4/train.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print('read data cost '+ str(time_2 - time_1)+' second')

    print('Start training')
    p = Softmax()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print('training cost '+ str(time_3 - time_2)+' second')

    print('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print('predicting cost ' + str(time_4 - time_3) +' second')

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is " + str(score))

    # 画图展示10个预测结果
    # for i in range(10):
    #     print(test_predict[i])
    #     print(test_labels[i])
    #     # 将展平的向量重新变成28*28的矩阵
    #     plt.imshow(test_features[i].reshape(28, 28))
    #     plt.show()

    # 画图展示10个数字的预测结果
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        # ax.imshow(X_test[i, 1:].reshape(28, 28), cmap='binary')
        # ax.set(title=f"Prediction: {np.argmax(y_pred, axis=1)[i]}")
        # ax.axis('off')
        ax.imshow(test_features[i].reshape(28, 28))
        ax.set(title=f"Prediction: {test_predict[i]}")
        ax.axis('off')

    plt.show()