from abc import ABC
import torch.nn as nn


class Classifier(nn.Module, ABC):
    def __init__(self):
        super(Classifier, self).__init__()

        def building_block(indim, outdim):
            return [
                nn.Conv2d(indim, outdim, 3, 1, 1),
                nn.BatchNorm2d(outdim),
                nn.ReLU(),
            ]

        def stack_blocks(indim, outdim, block_num):
            layers = building_block(indim, outdim)
            for i in range(block_num - 1):
                layers += building_block(outdim, outdim)
            layers.append(nn.MaxPool2d(2, 2, 0))
            return layers

        cnn_list = []
        cnn_list += stack_blocks(3, 128, 3)
        cnn_list += stack_blocks(128, 128, 3)
        cnn_list += stack_blocks(128, 256, 3)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential(*cnn_list)

        dnn_list = [
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 11),
        ]
        self.fc = nn.Sequential(*dnn_list)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape(out.size()[0], -1)
        return self.fc(out)
