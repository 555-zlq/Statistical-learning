from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图片并转换为灰度图片
img = Image.open('../../database/homework1/butterfly.bmp')
img = img.convert('L')

# 将图片转换为数组
img_arr = np.array(img)

# 将数据类型转换为float64
img_arr = img_arr.astype('float64')

# 均值化处理
mean = np.mean(img_arr, axis=0)
img_arr -= mean

# 将数据类型转换为uint8
img_arr = img_arr.astype('uint8')

# 计算协方差矩阵
cov = np.cov(img_arr.T)
cov = np.atleast_2d(cov)

# 计算特征向量和特征值
eig_val, eig_vec = np.linalg.eig(cov)

# 将特征值从大到小排序，并选择前N个最大特征值的特征向量
N = 250  # 选择前250个最大特征值
idx = eig_val.argsort()[::-1]  # 从大到小排序
eig_vec = eig_vec[:, idx]
eig_vec = eig_vec[:, :N]

# 计算降维后的矩阵
img_pca = np.dot(img_arr, eig_vec)

# 将降维后的矩阵转换为图片矩阵
img_pca = np.dot(img_pca, eig_vec.T) + mean
img_pca = np.reshape(img_pca, (img_arr.shape[0], img_arr.shape[1]))

# 将图像数据类型转换为float
img_pca = np.real(img_pca)  # 取实部
img_pca -= np.min(img_pca)
img_pca /= np.max(img_pca)
img_pca *= 255
img_pca = img_pca.astype('uint8')

# 绘制原始图片和降维后的图片
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_pca, cmap='gray')
plt.title('PCA Image')
plt.axis('off')

# 保存图片
Image.fromarray(img_pca).save('PCA_nopackage_butterfly.jpg')

plt.show()