import cv2
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans

color_dict = {
    "棕色": "brown",
    "黑色": "black",
    "红色": "red",
    "绿色": "green",
    "白色": "white"
}


def color_detect(h, s, v):
    if (0 <= h <= 180) and (0 <= s <= 255) and (0 <= v <= 46):
        return "black"
        # 红、紫
    elif (0 <= h <= 10 or 125 <= h <= 180) and (43 <= s <= 255) and (
            46 <= v <= 255):
        return "red"
        # 绿、青、蓝
    elif (35 <= h <= 124) and (43 <= s <= 255) and (46 <= v <= 255):
        return "green"
        # 白
    elif (0 <= h <= 180) and (0 <= s <= 30) and (221 <= v <= 255):
        return "white"
        # 灰
    elif (0 <= h <= 180) and (0 <= s <= 43) and (46 <= v <= 220):
        return "gray"
        # 橙、黄
    elif (11 <= h <= 34) and (43 <= s <= 255) and (46 <= v <= 255):
        return "brown"
    else:
        return "error"

class color_main():
    def colorfind(self,image_path):
        color_counts = {
            "black": 0,
            "red": 0,
            "green": 0,
            "white": 0,
            "gray": 0,
            "brown": 0,
            "error": 0,
        }

        image = cv2.imread(image_path)
        # 将图像转换为hsv格式
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 调整图像大小以加快聚类过程
        resized_image = cv2.resize(image_hsv, (50, 50))

        # 转换图像形状以适应K-Means输入
        pixels = resized_image.reshape((-1, 3))

        # 执行K-Means聚类
        num_clusters = 10  # 要聚类的颜色数目
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(pixels)

        # 获取聚类中心，即主要颜色
        main_colors = kmeans.cluster_centers_.astype(int)

        # 判断颜色，并计数
        for i in range(num_clusters):
            # print(main_colors[i])
            h, s, v = int(main_colors[i][0]), int(main_colors[i][1]), int(main_colors[i][2])
            color_class = color_detect(h, s, v)
            color_counts[color_class] += 1

        # 将灰色与其他颜色视为外值
        color_counts["gray"] = 0
        color_counts["error"] = 0

        # 得到出现次数最多的颜色
        best_color = max(color_counts, key=color_counts.get)
        return best_color






def save_to_output_file(image_path, res):
    with open(output_file, 'a') as f:
        f.write(f"{image_path} ")
        f.write(f"{res}\n")

#图片及标签制作
def labels_maker(animal_faeces_path):
    df = pd.read_excel(animal_faeces_path, sheet_name="宠物粪便")
    data = df.values

    for index in range(402):
        Mtype = str(data[index][10])
        data[index][10] =  color_dict[Mtype]

    img_name_list =[]
    label_list = []

    for i in range(402):
        img_name_list.append(data[i][6] + '.jpg')
        label_list.append(data[i][10])
    return img_name_list, label_list

if __name__ == '__main__':

    input_folder = "./new_data"
    label_path = "./label.txt"
    output_file = 'output.txt'

    open(output_file, 'w').close()


    image_files = []
    label_files = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            image, label = line.strip().split(" ")
            image_files.append(image)
            label_files.append(label)



    count = 0
    # # 遍历文件夹中的图像文件
    for filename, label in zip(image_files, label_files):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            print(filename)
            image_path = os.path.join(input_folder, filename)
            res = color_main().colorfind(image_path)
            if res == label:
                count += 1
            save_to_output_file(image_path, res)

    acc = count / 100.0
    print(acc)

