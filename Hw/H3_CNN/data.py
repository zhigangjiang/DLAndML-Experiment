import os
import torch
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImgDataset(Dataset):
    def __init__(self, img_dir, set_names, mode='train'):
        self.imgPaths = []
        self.labels = []
        for set_name in set_names:
            train_paths, train_labels = self.get_paths(os.path.join(img_dir, set_name))
            self.imgPaths.extend(train_paths)
            self.labels.extend(train_labels)

        if len(self.labels):
            if len(self.labels) != len(self.imgPaths):
                print('labels length:{} != imgPaths length:{}'.format(len(self.labels), len(self.imgPaths)))
                exit(0)

        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
            transforms.RandomRotation(15),  # 隨機旋轉圖片
            transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ])
        eval_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        self.transform = train_transforms if mode == 'train' or 'continue' else eval_transforms
        print("{} length: {}".format([set_name for set_name in set_names], self.__len__()))

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

    @staticmethod
    def get_paths(path):
        imgnames = os.listdir(path)
        imgnames.sort()
        imgpaths = []
        labels = []
        for name in imgnames:
            imgpaths.append(os.path.join(path, name))
            labels.append(int(name.split('_')[0]))
        labels = torch.LongTensor(labels)
        return imgpaths, labels

    def get_batch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)
