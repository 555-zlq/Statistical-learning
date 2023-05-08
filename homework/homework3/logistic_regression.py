# 用逻辑回归实现 MNIST 二分类

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np


# 定义 sigmoid 函数，用于计算激活值
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义损失函数，用于计算负对数似然
def neg_log_likelihood(theta, x, y):
    theta = theta.reshape(-1, 1)  # 转换为列向量
    y = y.reshape(-1, 1)  # 转换为列向量
    z = np.dot(x, theta)  # 计算激活值
    return np.mean(-y * z + np.log(1 + np.exp(z)))  # 计算损失


# 定义梯度函数，用于计算梯度
def grad(theta, x, y):
    theta = theta.reshape(-1, 1)  # 转换为列向量
    y = y.reshape(-1, 1)  # 转换为列向量
    z = np.dot(x, theta)  # 计算激活值
    return 1 / x.shape[0] * np.dot(x.T, sigmoid(z) - y)  # 计算梯度


# 加载训练数据
print('Loading training data...')
X_train, y_train = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X_train = X_train[(y_train == '0') | (y_train == '1')]  # 只取标签为0和1的数据
y_train = y_train[(y_train == '0') | (y_train == '1')]  # 只取标签为0和1的数据
scaler = StandardScaler()  # 初始化特征缩放器
X_train = scaler.fit_transform(X_train.astype(np.float32))  # 特征缩放
X_train = np.insert(X_train, 0, 1, axis=1)  # 插入常数项
y_train = y_train.reshape(-1, 1)  # 转换为列向量
y_train = y_train.astype(np.float32)  # 转换为浮点数

# 训练逻辑回归模型
theta0 = np.zeros((X_train.shape[1], 1))  # 初始化参数
learning_rate = 0.1  # 学习率
num_iterations = 500  # 迭代次数

print('Training...')
for i in range(num_iterations):
    theta0 -= learning_rate * grad(theta0, X_train, y_train)  # 梯度下降

# 加载测试数据
print('Loading testing data...')
X_test, y_test = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X_test = X_test[(y_test == '0') | (y_test == '1')]  # 只取标签为0和1的数据
y_test = y_test[(y_test == '0') | (y_test == '1')]  # 只取标签为0和1的数据
X_test = scaler.transform(X_test.astype(np.float32))  # 特征缩放
X_test = np.insert(X_test, 0, 1, axis=1)  # 插入常数项
y_test = y_test.reshape(-1, 1)  # 转换为列向量
y_test = y_test.astype(np.float32)  # 转换为浮点数

# 在训练集和测试集上进行预测
print('Predicting...')
y_train_pre = (sigmoid(np.dot(X_train, theta0)) >= 0.5).astype(int)  # 预测训练集
y_test_pre = (sigmoid(np.dot(X_test, theta0)) >= 0.5).astype(int)  # 预测测试集

# 计算分类准确度
train_accuracy = np.mean(y_train_pre == y_train)
test_accuracy = np.mean(y_test_pre == y_test)

print(f'Train accuracy: {train_accuracy:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')
