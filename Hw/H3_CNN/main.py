import os
import torch

from Hw.H3_CNN.model import CNN5
from Hw.H3_CNN.train import training
from Hw.H3_CNN.data import ImgDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import os

# import cv2
# import torch
# import torch.nn as nn
#
# import pandas as pd

# import time

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# process path
workspace_dir = '/Users/jiangzhigang/Google Drive/DataSet/food-11'
model_dir = "./model"

batch_size = 128
epoch = 30
lr = 0.001
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])
train_set = ImgDataset(os.path.join(workspace_dir, "training"), train_transforms)
print("train_set length: {}".format(train_set.__len__()))
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
val_set = ImgDataset(os.path.join(workspace_dir, "validation"), val_transforms)
print("val_set length: {}".format(val_set.__len__()))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = CNN5().cuda(device)

training(epoch, lr, model_dir, train_loader, val_loader, model, device)
