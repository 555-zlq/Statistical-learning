import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

# 读入原始图像，转换为灰度图像
img = Image.open('../../database/homework1/butterfly.bmp').convert('L')
img_data = np.array(img)

# 使用 PCA 进行图像压缩
pca = PCA(0.95)  # 保留 95% 的方差
img_compressed = pca.fit_transform(img_data)
img_reconstructed = pca.inverse_transform(img_compressed)

# 显示原始图像和经过压缩重建后的图像
img.show()
Image.fromarray(np.uint8(img_reconstructed)).show()

# 将图片保存为 jpg 格式
Image.fromarray(np.uint8(img_reconstructed)).save('PCA_butterfly.jpg')
