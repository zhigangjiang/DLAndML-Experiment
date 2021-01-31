import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import argparse
import ast

from Hw.H6_Adversarial_Attack.attacker import Attacker
from Hw.H6_Adversarial_Attack.data import AdverDataset

parser = argparse.ArgumentParser(usage="it's usage tip.", description="--h help info.")
parser.add_argument("--visible_device", default=0, type=int, help="visible device",
                    dest="visible_device")
parser.add_argument("--data_dir", required=True, type=str, help="the dataset root dir", dest="data_dir")
args = parser.parse_args()

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

data_dir = args.data_dir

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
if torch.cuda.is_available():
    visible_device = args.visible_device
    torch.cuda.set_device(visible_device)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device: {}".format(device))

########################################################################################################################

labels_csv_path = os.path.join(data_dir, "labels.csv")
label_name_csv_path = os.path.join(data_dir, "categories.csv")
# 讀入圖片相對應的 label
df = pd.read_csv(labels_csv_path)
df = (df.loc[:, 'TrueLabel']).to_numpy()
label_name = pd.read_csv(label_name_csv_path)
label_name = (label_name.loc[:, 'CategoryName']).to_numpy()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

model = models.vgg16(pretrained=True)
model.to(device)
model.eval()

dataset = AdverDataset(data_dir, ["images"], df, mean, std)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False)

attacker = Attacker(model, loader, mean, std, device)

epsilons = [0.1, 0.01]
accuracies, examples = [], []
# 進行攻擊 並存起正確率和攻擊成功的圖片
for eps in epsilons:
    ex, acc = attacker.attack(eps)
    accuracies.append(acc)
    examples.append(ex)

########################################################################################################################

cnt = 0
plt.figure(figsize=(30, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) * 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, orig_img, ex = examples[i][j]
        # plt.title("{} -> {}".format(orig, adv))
        plt.title("original: {}".format(label_name[orig].split(',')[0]))
        orig_img = np.transpose(orig_img, (1, 2, 0))
        plt.imshow(orig_img)
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) * 2, cnt)
        plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
        ex = np.transpose(ex, (1, 2, 0))
        plt.imshow(ex)
plt.tight_layout()
plt.show()
plt.save("result.jpg")