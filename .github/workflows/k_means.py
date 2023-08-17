import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('test/1_1_00005.jpg')

# 将图像转换为RGB格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 调整图像大小以加快聚类过程
resized_image = cv2.resize(image_rgb, (100, 100))

# 转换图像形状以适应K-Means输入
pixels = resized_image.reshape((-1, 3))

# 执行K-Means聚类
num_clusters = 5  # 要聚类的颜色数目
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# 获取聚类中心，即主要颜色
main_colors = kmeans.cluster_centers_.astype(int)

# 显示原始图像和主要颜色
plt.subplot(1, 2, 1)
plt.imshow(resized_image)
plt.title("origin image")
plt.axis('off')

plt.subplot(1, 2, 2)
colors = np.array(main_colors).reshape(1, -1, 3)
plt.imshow(colors)
plt.title("main color")
plt.axis('off')

plt.show()
