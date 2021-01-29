from Hw.H5_CNN_Explaination.data import ImgDataset
from Hw.H5_CNN_Explaination.utils import *
from Hw.H5_CNN_Explaination.model import Classifier
from Hw.H3_CNN.model import CNN5
from Hw.H3_CNN.utils import get_best_checkpoint_path, test, evaluate
import copy
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image

import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import argparse
import ast

parser = argparse.ArgumentParser(usage="it's usage tip.", description="--h help info.")
parser.add_argument("--mode", choices=['saliency', 'explaination', 'lime'], required=True, type=str,
                    help="saliency - saliency map; explaination - filter explaination; lime - Local Interpretable ModelAgnostic Explanations",
                    dest="mode")
parser.add_argument("--visible_device", default=0, type=int, help="visible device",
                    dest="visible_device")
parser.add_argument("--data_dir", required=True, type=str, help="the dataset root dir", dest="data_dir")
parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="the output checkpoints dir",
                    dest="checkpoint_dir")
parser.add_argument("--checkpoint_path", default="", type=str, help="the output checkpoints path",
                    dest="checkpoint_path")
args = parser.parse_args()

print("arguments:")
for arg in vars(args):
    print(arg, ":", getattr(args, arg))

print("-" * 100)

mode = args.mode
data_dir = args.data_dir
checkpoint_dir = args.checkpoint_dir
checkpoint_path = args.checkpoint_path

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
if torch.cuda.is_available():
    visible_device = args.visible_device
    torch.cuda.set_device(visible_device)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device: {}".format(device))

########################################################################################################################

model = Classifier().to(device)
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss

checkpoint_path = checkpoint_path if len(checkpoint_path) != 0 else get_best_checkpoint_path(checkpoint_dir)
print("load checkpoint_path:{}".format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)  # 加载断点
model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数

train_set = ImgDataset(data_dir, ["training", "validation"], "eval")

# 指定想要一起 visualize 的圖片 indices
img_indices = [830, 4218, 4707, 8598]
images, labels = train_set.get_batch(img_indices)

########################################################################################################################

if mode == "saliency":
    saliencies = compute_saliency_maps(images, labels, model, device)

    # 使用 matplotlib 畫出來
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            axs[row][column].imshow(img.permute(1, 2, 0).detach().numpy())
            # 小知識：permute 是什麼，為什麼這邊要用?
            # 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)
            # 但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
            # 因此 permute 是一個 pytorch 很方便的工具來做 dimension 間的轉換
            # 這邊 img.permute(1, 2, 0)，代表轉換後的 tensor，其
            # - 第 0 個 dimension 為原本 img 的第 1 個 dimension，也就是 height
            # - 第 1 個 dimension 為原本 img 的第 2 個 dimension，也就是 width
            # - 第 2 個 dimension 為原本 img 的第 0 個 dimension，也就是 channels

    plt.show()
    plt.close()

    # 從第二張圖片的 saliency，我們可以發現 model 有認出蛋黃的位置
    # 從第三、四張圖片的 saliency，雖然不知道 model 細部用食物的哪個位置判斷，但可以發現 model 找出了食物的大致輪廓

########################################################################################################################

if mode == "explaination":
    filter_activations, filter_visualizations = filter_explaination(copy.deepcopy(images), model, device=device,
                                                                    cnnid=34,
                                                                    filterid=0,
                                                                    iteration=10, lr=0.1)

    # 畫出 filter activations
    fig, axs = plt.subplots(3, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0).detach().numpy())
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    for i, img in enumerate(filter_visualizations):
        axs[2][i].imshow(normalize(img.permute(1, 2, 0).numpy()))

    plt.show()
    plt.close()
    # 根據圖片中的線條，可以猜測第 15 層 cnn 其第 0 個 filter 可能在認一些線條、甚至是 object boundary
    # 因此給 filter 看一堆對比強烈的線條，他會覺得有好多 boundary 可以 activate

########################################################################################################################

if mode == "lime":
    fig, axs = plt.subplots(1, 4, figsize=(15, 8))
    np.random.seed(16)
    # 讓實驗 reproducible
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        # lime 這個套件要吃 numpy array

        explainer = lime_image.LimeImageExplainer()

        def classifier_fn(input):
            return predict(input, model, device)
        explaination = explainer.explain_instance(image=x, classifier_fn=classifier_fn, segmentation_fn=segmentation)
        # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
        # classifier_fn 定義圖片如何經過 model 得到 prediction
        # segmentation_fn 定義如何把圖片做 segmentation
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

        lime_img, mask = explaination.get_image_and_mask(
            label=label.item(),
            positive_only=False,
            hide_rest=False,
            num_features=11,
            min_weight=0.05
        )
        # 把 explainer 解釋的結果轉成圖片
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask

        axs[idx].imshow(lime_img)

    plt.show()
    plt.close()
    # 從以下前三章圖可以看到，model 有認出食物的位置，並以該位置為主要的判斷依據
    # 唯一例外是第四張圖，看起來 model 似乎比較喜歡直接去認「碗」的形狀，來判斷該圖中屬於 soup 這個 class
    # 至於碗中的內容物被標成紅色，代表「單看碗中」的東西反而有礙辨認。
    # 當 model 只看碗中黃色的一坨圓形，而沒看到「碗」時，可能就會覺得是其他黃色圓形的食物。
    # lime 的笔记主要是每个图片输出对一个不同遮挡程度图预测的不同概率值，找一个线性model拟合这些输出概率，就会有一个权重值。