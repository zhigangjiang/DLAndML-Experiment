import torch.nn.functional as F
import torch.nn as nn
import os

import torch
import numpy as np
import pickle


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                    F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss


def knowledge_distillation_evaluate(epoch, teacher_model, student_model, loader, loss, device, optimizer, is_train):
    total_acc = 0.0
    total_loss = 0.0

    teacher_model.eval()
    if is_train:
        student_model.train()
        torch.set_grad_enabled(True)

    else:
        student_model.eval()  # fix BN and DropOut
        torch.set_grad_enabled(False)  # 和with torch.no_grad()一样 关闭grad计算，节省gpu空间

    for i, data in enumerate(loader):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        with torch.no_grad():
            soft_labels = teacher_model(inputs)

        outputs = student_model(inputs).squeeze()
        output_loss = loss(outputs, labels, soft_labels, 20, 0.5)

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


# #	name	    meaning	                    code	                            weight shape
# 0	cnn.{i}.0	Depthwise Convolution Layer	nn.Conv2d(x, x, 3, 1, 1, group=x)	(x, 1, 3, 3)
# 1	cnn.{i}.1	Batch Normalization	        nn.BatchNorm2d(x)	                (x)
# 2		        ReLU6	                    nn.ReLU6
# 3	cnn.{i}.3	Pointwise Convolution Layer	nn.Conv2d(x, y, 1),             	(y, x, 1, 1)
# 4		        MaxPooling	                nn.MaxPool2d(2, 2, 0)
def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()
    total = sum(p.numel() for p in old_model.parameters())
    trainable = sum(p.numel() for p in old_model.parameters() if p.requires_grad)
    print('old_model parameter total:{}, trainable:{}'.format(total, trainable))

    total = sum(p.numel() for p in new_model.parameters())
    trainable = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    print('new_model parameter total:{}, trainable:{}'.format(total, trainable))

    # selected_idx: 每一層所選擇的neuron index
    selected_idx = []
    # 我們總共有7層CNN，因此逐一抓取選擇的neuron index們。
    for i in range(8):
        # 根據上表，我們要抓的gamma係數在cnn.{i}.1.weight內。
        importance = params[f'cnn.{i}.1.weight']
        # 抓取總共要篩選幾個neuron。
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        # 以Ranking做Index排序，較大的會在前面(descending=True)。
        ranking = torch.argsort(importance, descending=True)
        # 把篩選結果放入selected_idx中。
        selected_idx.append(ranking[:new_dim])

    now_processed = 1
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        # 如果是cnn層，則移植參數。
        # 如果是FC層，或是該參數只有一個數字(例如batchnorm的tracenum等等資訊)，那麼就直接複製。
        if name.startswith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            # 當處理到Pointwise的weight時，讓now_processed+1，表示該層的移植已經完成。
            if name.startswith(f'cnn.{now_processed}.3'):
                now_processed += 1

            # 如果是pointwise，weight會被上一層的pruning和下一層的pruning所影響，因此需要特判。
            if name.endswith('3.weight'):
                # 如果是最後一層cnn，則輸出的neuron不需要prune掉。
                if len(selected_idx) == now_processed:
                    new_params[name] = p1[:, selected_idx[now_processed - 1]]
                # 反之，就依照上層和下層所選擇的index進行移植。
                # 這裡需要注意的是Conv2d(x,y,1)的weight shape是(y,x,1,1)，順序是反的。
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:, selected_idx[now_processed - 1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

    # 讓新model load進被我們篩選過的parameters，並回傳new_model。
    new_model.load_state_dict(new_params)
    return new_model


def encode16(params, fname):
    '''將params壓縮成16-bit後輸出到fname。

    Args:
      params: model的state_dict。
      fname: 壓縮後輸出的檔名。
    '''

    custom_dict = {}
    for (name, param) in params['net'].items():
        param = np.float64(param.cpu().numpy())
        # 有些東西不屬於ndarray，只是一個數字，這個時候我們就不用壓縮。
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param
    params['net'] = custom_dict
    pickle.dump(params, open(fname, 'wb'))


def decode16(fname):
    '''從fname讀取各個params，將其從16-bit還原回torch.tensor後存進state_dict內。

    Args:
      fname: 壓縮後的檔名。
    '''

    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params['net'].items():
        param = torch.tensor(param)
        custom_dict[name] = param
    params['net'] = custom_dict
    return params


def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params['net'].items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param
    params['net'] = custom_dict
    pickle.dump(params, open(fname, 'wb'))


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params['net'].items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param
    params['net'] = custom_dict
    return params
