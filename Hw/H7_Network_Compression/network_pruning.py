from Hw.H7_Network_Compression.data import ImgDataset
from Hw.H3_CNN.utils import get_best_checkpoint_path, test, evaluate
from Hw.H7_Network_Compression.utils import network_slimming
from Hw.H7_Network_Compression.mode import StudentNet
from Hw.H7_Network_Compression.train import network_pruning_training

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


# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

########################################################################################################################
model = StudentNet(base=16).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

# Pre-trained Network
checkpoint_path = checkpoint_path if len(checkpoint_path) != 0 else get_best_checkpoint_path(checkpoint_dir, "!pruned")
print("load checkpoint_path:{}".format(checkpoint_path))
checkpoint = torch.load(checkpoint_path)  # 加载断点
model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
start_epoch = checkpoint['epoch']  # 设置开始的epoch
best_acc = checkpoint['best_acc']

if mode != "train":
    checkpoint_path = checkpoint_path if len(checkpoint_path) != 0 else get_best_checkpoint_path(checkpoint_dir,
                                                                                                 "pruned")
    print("load checkpoint_path:{}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)  # 加载断点
    mult = checkpoint['mult']
    model = StudentNet(width_mult=mult).to(device)
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

        now_width_mult = 1
        for i in range(5):
            now_width_mult *= 0.95
            new_model = StudentNet(width_mult=now_width_mult).cuda()

            # Evaluate the importance and remove
            model = network_slimming(model, new_model)
            print("-" * 100)

            best_acc = 0
            # Fine-tune
            print("Fine-tune")
            network_pruning_training(0, 5, optimizer, checkpoint_dir, train_loader, val_loader,
                                     model, loss, best_acc,
                                     all_train,
                                     device, now_width_mult)
    else:
        val_loss, val_acc = evaluate(start_epoch, model, val_loader, loss, device, None,
                                     is_train=False)
