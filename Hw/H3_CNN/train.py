import torch
from torch import nn
import torch.optim as optim
import time
import numpy as np


def training(n_epoch, lr, model_dir, train_loader, val_loader, model, device):
    loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer 使用 Adam
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    best_acc = 0
    for epoch in range(n_epoch):
        epoch_start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):
            data_size = len(data[0])

            optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0])  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1])  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新參數值

            train_acc = np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy()) / data_size
            train_loss = batch_loss.item()
            total_acc += train_acc
            total_loss += train_loss

            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch + 1, i + 1, t_batch, train_loss, train_acc), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(
            total_loss / t_batch, total_acc / t_batch))

        total_acc = 0.0
        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data_size = len(data[0])
                val_pred = model(data[0])
                batch_loss = loss(val_pred, data[1])

                val_acc = np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy()) / data_size
                val_loss = batch_loss.item()
                total_acc += val_acc
                total_loss += val_loss

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(
                total_loss / v_batch, total_acc / v_batch))

            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir, total_acc/v_batch*100))
                # torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc / v_batch * 100))
            print('Time: {:.3f}\n-----------------------------------------------'.format(time.time() - epoch_start_time))
