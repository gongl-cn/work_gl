#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @File   : pet_Feces_Color_Lgn.py

import cv2
import numpy as np
import os
import pandas as pd

color_dict = {
    "棕色": "brown",
    "黑色": "black",
    "红色": "red",
    "绿色": "green",
    "白色": "white"
}

class color_main():
    def colorfind(self,image_path):
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        # 裁剪图片
        img_crop = img[int(height / 2):int(height / 2) + 150, int(width / 2) :int(width / 2) + 150]
        # cv2.imshow("img", img_crop)
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        # hist_var = np.mean(hsv, axis=(0, 1))

        # 求最大值
        hist_var = []
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
            hist_max = np.where(hist == np.max(hist))
            hist_var.append(hist_max[0])

        '''
        while True:
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27 or cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        '''
        # print(hist_var)
        hist_var_0, hist_var_1, hist_var_2 = int(hist_var[0][0]), int(hist_var[1][0]), int(hist_var[2][0])
        # 黑
        if (0 <= hist_var_0 <= 180) and (0 <= hist_var_1 <= 255) and (0 <= hist_var_2 <= 46):
            return "black"
        # 红、紫
        elif (0 <= hist_var_0 <= 10 or 125 <= hist_var_0 <= 180) and (43 <= hist_var_1 <= 255) and (
                46 <= hist_var_2 <= 255):
            return "red"
        # 绿、青、蓝
        elif (35 <= hist_var_0 <= 124) and (43 <= hist_var_1 <= 255) and (46 <= hist_var_2 <= 255):
            return "green"
        # 白
        elif (0 <= hist_var_0 <= 180) and (0 <= hist_var_1 <= 30) and (221 <= hist_var_2 <= 255):
            return "white"
        # 灰
        elif (0 <= hist_var_0 <= 180) and (0 <= hist_var_1 <= 43) and (46 <= hist_var_2 <= 220):
            return "gray"
        # 橙、黄
        elif (11 <= hist_var_0 <= 34) and (43 <= hist_var_1 <= 255) and (46 <= hist_var_2 <= 255):
            return "brown"
        else:
            return "其它"

        # 求均值
        # h, s, v = cv2.split(hsv)
        # h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)

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
    # animal_faeces_path = "/home/gonglei/data/animal_faeces_cla/宠物智能识别系统信息采集_.xlsx"
    input_folder = "/home/gonglei/data/animal_faeces_cla/crop_data/data/"
    label_path = "/home/gonglei/code/classify/classify/color_classify/labels_part.txt"
    output_file = '/home/gonglei/data/animal_faeces_cla/output_.txt'

    # 清空已有的输出文件内容
    open(output_file, 'w').close()

    # 获取图片及标签数据
    # images, labels = labels_maker(animal_faeces_path)
    # fileName = "labels.txt"
    # for image, label in zip(images, labels):
    #     with open(fileName, 'a') as file:
    #         file.write(image + " " + label + "\n")

    image_files = []
    label_files = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            image, label = line.strip().split(" ")
            image_files.append(image)
            label_files.append(label)
    dataset_length = len(image_files)
    
    # image_files = [filename for filename in os.listdir(input_folder) if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    #
    # image_files.sort()

    count = 0
    # # 遍历文件夹中的图像文件
    for filename, label in zip(image_files, label_files):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(input_folder, filename)
            res = color_main().colorfind(image_path)
            if res == label:
                count += 1
            save_to_output_file(image_path, res)

    acc = count / dataset_length
    print(acc)

    # image_path = "/home/gonglei/data/animal_faeces_cla/crop_data/data/1_1_00320.jpg"
    # res = color_main().colorfind(image_path)
    # print(res)
