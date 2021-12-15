# coding=utf-8
"""
@author: Cao Zhanxiang
@project: project 2
@file: Application.py
@date: 2021/12/9
@function: application of the trained model
"""

from cv2 import cv2
from torch import nn
import torch
from torchvision import transforms as F
from Model import ERNet
import numpy as np
import argparse
import os

# 命令行参数
parser = argparse.ArgumentParser(description='to outline the face & predict the expression in a picture/video')
parser.add_argument('--path', type=str, default='./data/img.jpg', help='the picture/video file path, press Q can stop the playing of video. <Advised options>: [--path ./data/img.jpg] or [--path ./data/video.mp4]')
args = parser.parse_args()

# predict -> label name
Dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def ImgPred(face_model, express_model, img):
    """
    outline the face in img and label the face expression

    :param face_model: the trained model to predict the expression
    :param express_model: the cv2 trained model to find the location of the face
    :param img: the input img
    :return: the img with bounding box & label
    """
    # 转换为灰度图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 获取人脸位置
    faces_loc = face_model.detectMultiScale(gray_img, 1.1, 3)

    for (x, y, w, h) in faces_loc:
        '''预测表情'''
        # 截取灰度图
        face_img = gray_img[y: y + h, x: x + w]
        # resize & ToTensor & expand dim to 1 x 1 x 48 x 48
        input_img = cv2.resize(face_img, (48, 48))
        input_img = F.ToTensor()(input_img)
        input_img = torch.unsqueeze(input_img, 0)
        # into the model
        output = express_model(input_img)
        pred = torch.argmax(output, 1)
        label = Dict[pred.item()]
        print(label)
        '''绘制bbox'''
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        '''绘制label'''
        # 设置字体格式及大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 获取label长宽
        label_size = cv2.getTextSize(label, font, 1, 2)
        # 设置label起点
        text_origin1 = np.array([x, y - label_size[0][1]])
        # thickness=-1颜色填充
        cv2.rectangle(img, tuple(text_origin1), tuple(text_origin1 + label_size[0]), color=(0, 255, 0), thickness=-1)
        # label text
        cv2.putText(img, label, (x, y), font, fontScale=1, color=(255, 255, 255), thickness=2)

    # resize the height = 500
    ratio = (5000 // img.shape[0]) / 10
    img = cv2.resize(img, None, fx=ratio, fy=ratio)

    return img


if __name__ == '__main__':
    # 加载人脸识别模型
    FaceReco = cv2.CascadeClassifier('./trained/haarcascade_frontalface_default.xml')
    # 加载表情识别模型
    ExpressModel = torch.load('./trained/max.pkl', map_location=torch.device(DEVICE))

    if not os.path.exists(args.path):
        print('invalid file path')

    try:
        if args.path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg', 'gif', 'bmp']:
            '''参数为图片类型'''
            # 读取图片
            Img = cv2.imread(args.path)
            # 获取预测结果图片
            InfoImg = ImgPred(FaceReco, ExpressModel, Img)
            # 显示图片
            cv2.imshow('Img', InfoImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif args.path.split('.')[-1].lower() in ['avi', 'rmvb', 'rm', 'mp4', 'flv', 'mpg']:
            '''参数为视频类型'''
            # 读取视频
            cap = cv2.VideoCapture(args.path)
            # predict
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                InfoImg = ImgPred(FaceReco, ExpressModel, frame)
                cv2.imshow('img', InfoImg)
                # Q 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
    except cv2.error:
        print('invalid file path')
