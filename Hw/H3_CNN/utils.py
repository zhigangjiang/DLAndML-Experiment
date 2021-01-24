import os

import torch
import numpy as np


def get_best_checkpoint_path(checkpoint_dir):
    best_mode_path = os.path.join(checkpoint_dir, sorted([file_name for file_name in os.listdir(checkpoint_dir)
                                                          if file_name.__contains__(".pth")],
                                                         key=lambda file_name: float(
                                                             file_name.split("_")[-1].split(".pth")[0]),
                                                         reverse=True)[0])
    return best_mode_path


def evaluate(epoch, model, loader, loss, device, optimizer, is_train):
    total_acc = 0.0
    total_loss = 0.0

    if is_train:
        model.train()
        torch.set_grad_enabled(True)

    else:
        model.eval()
        torch.set_grad_enabled(False)

    for i, data in enumerate(loader):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(inputs).squeeze()
        output_loss = loss(outputs, labels)

        if is_train and optimizer:
            optimizer.zero_grad()
            output_loss.backward()
            optimizer.step()

        batch_acc = np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == data[1].numpy()) / len(data[0])
        batch_loss = output_loss.item()

        total_acc += batch_acc
        total_loss += batch_loss
        print('\r[ Epoch{}: {}/{} - {}] loss:{:.3f} acc:{:.3f} '.format(
            epoch + 1, i + 1, len(loader), "train" if is_train else "eval", batch_loss, batch_acc), end='', flush=True)

    mean_acc = total_acc / len(loader)
    mean_loss = total_loss / len(loader)
    print('\n{} | Loss:{:.5f} Acc: {:.3f}'.format("train" if is_train else "eval", mean_loss, mean_acc))
    return mean_loss, mean_acc


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
