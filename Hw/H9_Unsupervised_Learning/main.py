import torch.nn as nn
from torch import optim
from Hw.H9_Unsupervised_Learning.utils import *
from Hw.H9_Unsupervised_Learning.model import AE
from Hw.H9_Unsupervised_Learning.data import Image_Dataset
from Hw.H9_Unsupervised_Learning.train import *
from torch.utils.data import DataLoader

import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import argparse
import ast

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--mode", choices=['train', 'continue', 'evaluate', 'test', 'inference'], required=True, type=str,
                    help="the run mode", dest="mode")
parser.add_argument("--visible_device", default=0, type=int, help="visible device",
                    dest="visible_device")
parser.add_argument("--data_dir", required=True, type=str, help="the dataset root dir", dest="data_dir")
parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="the output checkpoints dir",
                    dest="checkpoint_dir")
parser.add_argument("--lr", default=1e-5, type=float, help="the learning rate of adama.", dest="lr")
parser.add_argument("--batch_size", default=128, type=int, help="batch_size", dest="batch_size")
parser.add_argument("--checkpoint_path", default="", type=str, help="the output checkpoints path",
                    dest="checkpoint_path")
args = parser.parse_args()

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

mode = args.mode
data_dir = args.data_dir
visible_device = args.visible_device

lr = float(args.lr)
batch_size = int(args.batch_size)
checkpoint_dir = args.checkpoint_dir
checkpoint_path = args.checkpoint_path
start_epoch = 0
epoch = 500
best_loss = 9999999

torch.cuda.set_device(visible_device)

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if mode == "train" and len([f for f in os.listdir(checkpoint_dir) if f.__contains__(".pth")]) > 0:
    ans = input("checkpoints is not empty, do you cover? [y/n]")
    if ans != 'y':
        exit(0)

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

########################################################################################################################

same_seeds(0)

model = AE().cuda()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

if mode != "train":
    checkpoint_path = checkpoint_path if len(checkpoint_path) != 0 else get_best_checkpoint_path(checkpoint_dir)
    print("load checkpoint_path:{}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    best_loss = checkpoint['epoch_loss']

trainX = None
valX = None
valY = None
if mode != 'test':
    trainX = np.load(os.path.join(data_dir, 'trainX.npy'))
    trainX_preprocessed = preprocess(trainX)
    img_dataset = Image_Dataset(trainX_preprocessed)
    # 準備 dataloader, model, loss criterion 和 optimizer
    img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True if mode == 'train' else False)
else:
    valX = np.load(os.path.join(data_dir, 'valX.npy'))
    valY = np.load(os.path.join(data_dir, 'valY.npy'))
    valX_preprocessed = preprocess(valX)
    img_dataset = Image_Dataset(valX_preprocessed)
    # 準備 dataloader, model, loss criterion 和 optimizer
    img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False)

if mode == "train" or mode == "continue":
    train(start_epoch, epoch, model, img_dataloader, loss, optimizer, best_loss, checkpoint_dir, device)

else:
    # 預測答案
    latents = inference(img_dataloader, model, device)
    pred, X_embedded = predict(latents)
    if mode == "inference":
        # 將預測結果存檔，上傳 kaggle
        save_prediction(pred, os.path.join(checkpoint_dir, 'prediction.csv'))
        # 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
        # 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
        save_prediction(invert(pred), os.path.join(checkpoint_dir, 'prediction_invert.csv'))
    elif mode == 'test':
        acc_latent = cal_acc(valY, pred)
        print('The clustering accuracy is:', acc_latent)
        print('The clustering result:')
        plot_scatter(X_embedded, valY, savefig=os.path.join(checkpoint_dir, 'p1_baseline.png'))

