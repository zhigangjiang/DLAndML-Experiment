import os
import torch
import cv2
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.imgNames = os.listdir(img_dir)
        # label is required to be a LongTensor
        self.Labels = [int(imgName.split("_")[0]) for imgName in self.imgNames]
        if len(self.Labels):
            self.Labels = torch.LongTensor(self.Labels)
        self.transform = transform
        self.imgDir = img_dir

    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, index):
        # print(index)
        # print(os.path.join(self.imgDir, self.imgNames[index]))
        img = cv2.imread(os.path.join(self.imgDir, self.imgNames[index]))
        x = cv2.resize(img, (128, 128))

        if self.transform is not None:
            x = self.transform(x)
        if len(self.Labels):
            y = self.Labels[index]
            return x, y
        else:
            return x
