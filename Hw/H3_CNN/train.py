import torch
from torch import nn
import time
import numpy as np
import os
from Hw.H3_CNN.test import evaluate, backward


def training(star_epoch, n_epoch, optimizer, checkpoint_dir, train_loader, val_loader, model, loss, all_train, device):
    best_acc = 0
    for epoch in range(star_epoch, n_epoch):
        print("epoch:{}".format(epoch))

        epoch_start_time = time.time()
        train_loss, train_acc = backward(epoch, model, train_loader, loss, optimizer, device)
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(train_loss, train_acc))

        if not all_train:
            val_loss, val_acc = evaluate(model, val_loader, loss, device)
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(val_loss, val_acc))

            if val_acc <= best_acc:
                continue
            best_acc = val_acc
        else:
            if train_acc <= best_acc:
                continue
            best_acc = train_acc

        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint,
                   os.path.join(checkpoint_dir, "epoch_{}_val_acc_{:.3f}.pth".format(epoch, best_acc)))
        # torch.save(model, "{}/ckpt.model".format(model_dir))
        print('saving model with acc {:.3f}'.format(best_acc))
        print('Time: {:.3f}\n-----------------------------------------------'.format(time.time() - epoch_start_time))
