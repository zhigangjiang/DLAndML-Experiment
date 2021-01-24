import os
import torch

import torch
from torch import nn
import time
import numpy as np
import os


def evaluate(model, val_loader, loss, device):
    total_acc = 0.0
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data_size = len(data[0])
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc = np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy()) / data_size
            val_loss = batch_loss.item()
            total_acc += val_acc
            total_loss += val_loss

    val_acc = total_acc / len(val_loader)
    val_loss = total_loss / len(val_loader)
    return val_loss, val_acc


def backward(epoch, model, train_loader, loss, optimizer, device):
    total_acc = 0.0
    total_loss = 0.0
    model.train()

    for i, data in enumerate(train_loader):
        data_size = len(data[0])

        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].to(device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc = np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy()) / data_size
        train_loss = batch_loss.item()
        total_acc += train_acc
        total_loss += train_loss

        print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            epoch + 1, i + 1, len(train_loader), train_loss, train_acc), end='\r')

    train_acc = total_acc / len(train_loader)
    train_loss = total_loss / len(train_loader)
    return train_loss, train_acc


def test(model, test_loader, device):
    model.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            print(test_label)
            for y in test_label:
                prediction.append(y)
    return prediction
