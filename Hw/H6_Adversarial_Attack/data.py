import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class AdverDataset(Dataset):
    def __init__(self, img_dir, set_names, labels, mean, std):
        self.imgPaths = []
        self.labels = torch.from_numpy(labels).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        for set_name in set_names:
            train_paths = self.get_paths(os.path.join(img_dir, set_name))
            self.imgPaths.extend(train_paths)

        # 为什么Pytorch用mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]来正则化
        # 这是因为使用了使用ImageNet的均值和标准差。
        # 使用Imagenet的均值和标准差是一种常见的做法。它们是根据数百万张图像计算得出的。如果要在自己的数据集上从头开始训练，则可以计算新的均值和标准差。否则，建议使用Imagenet预设模型及其平均值和标准差。
        # mean和std对应是RGB通道 此外： PIL-RGB CV-BGR PLT-RGB
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            self.normalize
        ])

        self.transforms = transform

        print("{} length: {}".format([set_name for set_name in set_names], self.__len__()))

    def __getitem__(self, index):
        # 利用路徑讀取圖片
        img = Image.open(self.imgPaths[index])
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.imgPaths)

    @staticmethod
    def get_paths(path):
        imgnames = os.listdir(path)
        imgnames.sort()
        imgpaths = []
        for name in imgnames:
            if name.startswith('.'):
                continue
            imgpaths.append(os.path.join(path, name))
        return imgpaths
