import os
import torch

from Hw.H3_CNN.model import CNN5
from Hw.H3_CNN.train import training
from Hw.H3_CNN.data import ImgDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse
import ast


parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--data_dir", required=True, type=str, help="the dataset root dir", dest="data_dir")
parser.add_argument("--model_dir", default="./model", type=str, help="the output model dir", dest="model_dir")
parser.add_argument("--lr", default=0.001, type=float, help="the learning rate of adama.", dest="lr")
parser.add_argument("--batch_size", default=128, type=int, help="batch_size", dest="batch_size")
args = parser.parse_args()

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

data_dir = args.data_dir
lr = float(args.lr)
batch_size = int(args.batch_size)
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

epoch = 500


# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])
train_set = ImgDataset(os.path.join(data_dir, "training"), train_transforms)
print("train_set length: {}".format(train_set.__len__()))
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
val_set = ImgDataset(os.path.join(data_dir, "validation"), val_transforms)
print("val_set length: {}".format(val_set.__len__()))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = CNN5().to(device)

training(epoch, lr, model_dir, train_loader, val_loader, model, device)
