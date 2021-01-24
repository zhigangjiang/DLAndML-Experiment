# 這個 block 用來先定義一些等等常用到的函式
import torch
import os


def load_training_data(path='training_label.txt'):
    # 把 training 時需要的 data 讀進來
    # 如果是 'training_label.txt'，需要讀取 label，如果是 'training_nolabel.txt'，不需要讀取 label
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


def load_testing_data(path='testing_data'):
    # 把 testing 時需要的 data 讀進來
    with open(path, 'r') as f:
        lines = f.readlines()
        x = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        x = [sen.split(' ') for sen in x]
    return x


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
        inputs = data[0].to(device, dtype=torch.long)
        labels = data[1].to(device, dtype=torch.float)

        outputs = model(inputs).squeeze()
        output_loss = loss(outputs, labels)

        if is_train and optimizer:
            optimizer.zero_grad()
            output_loss.backward()
            optimizer.step()

        outputs[outputs >= 0.5] = 1  # 大於等於 0.5 為正面
        outputs[outputs < 0.5] = 0  # 小於 0.5 為負面

        batch_acc = torch.sum(torch.eq(outputs, labels)).item() / len(data[0])
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
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大於等於 0.5 為正面
            outputs[outputs < 0.5] = 0  # 小於 0.5 為負面
            ret_output += outputs.int().tolist()

    return ret_output
