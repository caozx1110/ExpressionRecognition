# coding=utf-8
"""
@author: Cao Zhanxiang
@project: project 2
@file: Eval.py
@date: 2021/12/11
@function: eval the trained model
"""

import torch
import numpy as np
from Model import ERNet
from sklearn.metrics import *
import matplotlib.pyplot as plt
from DataLoader import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def Eval(model_file, test_loader):
    """
    evaluate the model

    :param model_file: the path of the trained model
    :param test_loader: the data loader of test
    :return: None
    """
    model = torch.load(model_file, map_location=torch.device(DEVICE))
    # mode
    model.eval()
    # label and pred
    label_all = np.array([])
    pred_all = np.array([])

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            pred = torch.argmax(output, dim=1)
            # append
            label_all = np.append(label_all, label.cpu().numpy())
            pred_all = np.append(pred_all, pred.cpu().numpy())

    cm = confusion_matrix(label_all, pred_all)
    labels_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # ConfusionMatrix
    ConfusionMatrixDisplay(cm, display_labels=labels_name).plot(xticks_rotation=45)
    print('accuracy_score:', accuracy_score(label_all, pred_all))
    print('balanced_accuracy_score', balanced_accuracy_score(label_all, pred_all))
    print('classification_report', classification_report(label_all, pred_all, target_names=labels_name, digits=3))


if __name__ == '__main__':
    # load data
    test_trans = F.ToTensor()
    _, data_test = ReturnDataSet('./data/表情识别.csv', test_trans, test_trans)
    TestLoader = DataLoader(data_test, batch_size=64)

    Eval('./trained/max.pkl', TestLoader)


