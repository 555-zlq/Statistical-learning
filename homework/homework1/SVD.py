import numpy as np
from PIL import Image

# 读入原始图像，转换为灰度图像
img = Image.open('../../database/homework1/butterfly.bmp').convert('L')
img_data = np.array(img)

# 使用 SVD 进行图像的分解
U, Sigma, V = np.linalg.svd(img_data)

# 按照需要保留的奇异值个数，进行矩阵截断
# 这里保留前 200 个奇异值
k = 200
U_truncated = U[:, :k] # 取矩阵的前 k 列
Sigma_truncated = np.diag(Sigma[:k]) # 取矩阵的前 k 个奇异值
V_truncated = V[:k, :] # 取矩阵的前 k 行
img_compressed = U_truncated @ Sigma_truncated @ V_truncated # 重构图像
img_reconstructed = np.uint8(img_compressed) # 转换为 uint8 类型

# 显示原始图像和经过压缩重建后的图像
img.show()
Image.fromarray(np.uint8(img_reconstructed)).show()

# 将图片保存为 jpg 格式
Image.fromarray(np.uint8(img_reconstructed)).save('SVD_butterfly.jpg')