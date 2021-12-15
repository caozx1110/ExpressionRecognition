# coding=utf-8
"""
@author: Cao Zhanxiang
@project: project 2
@file: DataLoader.py
@date: 2021/12/9
@function: load the csv data
"""
import numpy as np
from csv import reader
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as F

def LoadData(filename):
    """
    load the np.ndarray from the csv file

    :param filename: the csv file name
    :return np.ndarray: the input & label 0f train & test
    """
    with open(filename, 'rt', encoding='UTF-8') as raw_data:
        readers = reader(raw_data)
        data = np.array(list(readers))
    # data.shape: (35888, 3)
    # train
    data_train_raw = data[data[:, 2] == 'Training']
    input_train = np.array([np.array(list(map(int, data_train_raw[i][1].strip().split()))).reshape(48, 48) 
                            for i in range(len(data_train_raw))])
    label_train = np.array([int(data_train_raw[i][0])
                            for i in range(len(data_train_raw))])
    # test
    data_test_raw = data[data[:, 2] == 'Test']
    input_test = np.array([np.array(list(map(int, data_test_raw[i][1].strip().split()))).reshape(48, 48)
                            for i in range(len(data_test_raw))])
    label_test = np.array([int(data_test_raw[i][0])
                            for i in range(len(data_test_raw))])
    return input_train, label_train, input_test, label_test

class MyDataSet(Dataset):
    """rewrite the DataSet"""
    def __init__(self, img, label, transform=None):
        super(MyDataSet, self).__init__()
        self.img = img
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.img[index]))
        if self.transform is not None:
            img = self.transform(img)
        label = self.label[index]
        img = np.asarray(img)
        # img = np.expand_dims(img, 0)

        return img, label

    def __len__(self):
        return len(self.img)

def ReturnDataSet(filename, train_trans, test_trans):
    """
    load dataset from the csv file

    :param filename: the csv file name
    :param train_trans: train transform
    :param test_trans: test transform
    :return: the train dataset & test dataset
    """
    # Load Data
    InputTrain, LabelTrain, InputTest, LabelTest = LoadData(filename)

    # data
    data_train = MyDataSet(InputTrain, LabelTrain, train_trans)
    # print(len(data_train))
    data_test = MyDataSet(InputTest, LabelTest, test_trans)
    # print(len(data_test))

    return data_train, data_test


if __name__ == '__main__':
    # train transform
    train_trans = F.Compose([
        F.RandomApply([F.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        F.RandomHorizontalFlip(),
        F.RandomApply([F.RandomRotation(10)], p=0.5),
        F.RandomCrop(48, 4),
        F.ToTensor(),
    ])
    # test transform
    test_trans = F.Compose([
        F.ToTensor(),
    ])

    data_train, data_test = ReturnDataSet('./data/表情识别.csv', train_trans, test_trans)
