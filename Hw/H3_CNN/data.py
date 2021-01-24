import os
import torch
import cv2
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        self.imgPaths = []
        for img_dir in img_dirs:
            self.imgPaths.extend([os.path.join(img_dir, imgName) for imgName in sorted(os.listdir(img_dir))])
            # label is required to be a LongTensor
        self.labels = [int(imgPath.split('/')[-1].split("_")[0]) for imgPath in self.imgPaths if
                       not imgPath.split('/')[-1].startswith('.') and len(imgPath.split('/')[-1].split("_")) > 1]
        if len(self.labels):
            self.labels = torch.LongTensor(self.labels)
            if len(self.labels) != len(self.imgPaths):
                print('labels length:{} != imgPaths length:{}'.format(len(self.labels), len(self.imgPaths)))
                exit(0)
        self.transform = transform
        print("{} length: {}".format([img_dir.split('/')[-1] for img_dir in img_dirs], self.__len__()))

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):

        img = cv2.imread(self.imgPaths[index])
        x = cv2.resize(img, (128, 128))
        if self.transform is not None:
            x = self.transform(x)
        if len(self.labels):
            y = self.labels[index]
            return x, y
        else:
            return x
