from Hw.H3_CNN.model import CNN5
from Hw.H3_CNN.utils import show_model_parameter_number
from Hw.H7_Network_Compression.data import ImgDataset
from Hw.H3_CNN.utils import get_best_checkpoint_path, test
from Hw.H7_Network_Compression.utils import knowledge_distillation_evaluate, loss_fn_kd, decode8, decode16
from Hw.H7_Network_Compression.mode import StudentNet
from Hw.H7_Network_Compression.train import knowledge_distillation_training

import torchvision.models as models
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
teacher_model = models.resnet18(pretrained=False, num_classes=11).cuda()
teacher_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'teacher_resnet18.bin')))
student_model = StudentNet(base=16).cuda()
optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
loss = loss_fn_kd

if mode != "train":
    checkpoint_path = checkpoint_path if len(checkpoint_path) != 0 else get_best_checkpoint_path(checkpoint_dir)
    print("load checkpoint_path:{}".format(checkpoint_path))

    if checkpoint_path.__contains__('encode16'):
        checkpoint = decode16(checkpoint_path)
    elif checkpoint_path.__contains__('encode8'):
        checkpoint = decode8(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path)  # 加载断点

    if checkpoint_path.__contains__('bin'):
        student_model.load_state_dict(checkpoint)
    else:
        student_model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        best_acc = checkpoint['best_acc']

if mode == "test":
    test_set = ImgDataset(data_dir, ["testing"], mode)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    prediction = test(student_model, test_loader, device)
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
        knowledge_distillation_training(start_epoch, epoch, optimizer, checkpoint_dir, train_loader, val_loader,
                                        teacher_model, student_model, loss, best_acc,
                                        all_train,
                                        device)
    else:
        print("student_model:")
        show_model_parameter_number(student_model, "student_model")
        knowledge_distillation_evaluate(start_epoch, teacher_model, student_model, val_loader, loss,
                                        device, None,
                                        is_train=False)
        print("teacher_model:")
        show_model_parameter_number(teacher_model, "teacher_model")
        knowledge_distillation_evaluate(start_epoch, student_model, teacher_model, val_loader, loss,
                                        device, None,
                                        is_train=False)
