# 梯度下降的线性回归，利用boston房价数据集

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression


# 加载波士顿房价数据集
boston = fetch_openml(name='boston', version=1)
X = boston.data
y = boston.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 初始化模型
regressor = LinearRegression(learning_rate=0.01, n_iterations=1000)

# 训练模型
regressor.fit(X_train, y_train)

# 预测测试集
predicted = regressor.predict(X_test)

# 计算R^2
def r2_score(y_true, y_predicted):
    corr_matrix = np.corrcoef(y_true, y_predicted)
    corr = corr_matrix[0, 1]
    return corr ** 2

r2 = r2_score(y_test, predicted)
print("R^2:", r2)

# 绘制图像，观察预测值与真实值的关系，以及残差，可以看出残差基本为0，说明模型拟合效果较好
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicted)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
#
plt.figure(figsize=(8, 6))
plt.scatter(predicted, y_test - predicted)
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()

# 绘制图像，横坐标为0-测试集样本数，纵坐标为预测值与真实值，从小到大排序, 用×表示真实值，用o表示预测值

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), sorted(y_test), label='true')
plt.scatter(range(len(y_test)), sorted(predicted), label='predicted')
plt.legend()
plt.show()










