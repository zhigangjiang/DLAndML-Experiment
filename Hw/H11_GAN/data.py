from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torchvision.transforms as transforms

import glob


class FaceDataset(Dataset):
    def __init__(self, data_dir):
        # resize the image to (64, 64)
        # linearly map [0, 1] to [-1, 1]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

        self.fnames = glob.glob(os.path.join(data_dir, '*'))
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img)  # because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    @staticmethod
    def BGR2RGB(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
