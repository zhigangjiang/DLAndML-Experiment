
from Hw.H3_CNN.model import CNN5
from Hw.H3_CNN.train import training
from Hw.H3_CNN.data import ImgDataset
from Hw.H3_CNN.utils import get_best_checkpoint_path, test, evaluate


import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import argparse
import ast

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--mode", choices=['train', 'continue', 'evaluate', 'test'], required=True, type=str,
                    help="the run mode", dest="mode")
parser.add_argument("--all_train", default=False, type=ast.literal_eval, help="is train dataset: train and evaluate.",
                    dest="all_train")
parser.add_argument("--visible_device", default=0, type=int, help="visible device",
                    dest="visible_device")
parser.add_argument("--data_dir", required=True, type=str, help="the dataset root dir", dest="data_dir")
parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="the output checkpoints dir",
                    dest="checkpoint_dir")
parser.add_argument("--lr", default=0.001, type=float, help="the learning rate of adama.", dest="lr")
parser.add_argument("--batch_size", default=128, type=int, help="batch_size", dest="batch_size")
parser.add_argument("--checkpoint_path", default="", type=str, help="the output checkpoints path",
                    dest="checkpoint_path")
args = parser.parse_args()

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

mode = args.mode
all_train = args.all_train
data_dir = args.data_dir
visible_device = args.visible_device

lr = float(args.lr)
batch_size = int(args.batch_size)
checkpoint_dir = args.checkpoint_dir
checkpoint_path = args.checkpoint_path
start_epoch = 0
epoch = 500
best_acc = 0

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

model = CNN5().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer 使用 Adam
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss

if mode != "train":
    checkpoint_path = checkpoint_path if len(checkpoint_path) != 0 else get_best_checkpoint_path(checkpoint_dir)
    print("load checkpoint_path:{}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    best_acc = checkpoint['best_acc']

if mode == "test":
    test_set = ImgDataset(data_dir, ["testing"], mode)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    prediction = test(model, test_loader, device)
    with open(os.path.join(checkpoint_dir, "predict.csv"), 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            print(y)
            f.write('{},{}\n'.format(i, y))

else:

    val_set = ImgDataset(data_dir, ["validation"], mode)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    if mode == "train" or mode == "continue":
        train_set = ImgDataset(data_dir, ["training", "validation"] if all_train else ["training"], mode)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        training(start_epoch, epoch, optimizer, checkpoint_dir, train_loader, val_loader, model, loss, best_acc,
                 all_train,
                 device)
    else:
        val_loss, val_acc = evaluate(start_epoch, model, val_loader, loss, device, None, is_train=False)
