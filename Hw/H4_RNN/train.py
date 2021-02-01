import torch
import time
import os
from Hw.H4_RNN.utils import evaluate
from Hw.H3_CNN.utils import show_model_parameter_number


def training(star_epoch, n_epoch, optimizer, checkpoint_dir, train_loader, val_loader, model, loss, best_acc, all_train,
             device):
    show_model_parameter_number(model, "model")

    for epoch in range(star_epoch, n_epoch):
        print("-" * 100)
        print("epoch:{} best_acc:{}".format(epoch + 1, best_acc))

        epoch_start_time = time.time()
        train_loss, train_acc = evaluate(epoch, model, train_loader, loss, device, optimizer, is_train=True)
        cur_acc = train_acc

        if not all_train:
            val_loss, val_acc = evaluate(epoch, model, val_loader, loss, device, None, is_train=False)
            cur_acc = val_acc

        if cur_acc > best_acc:
            best_acc = cur_acc
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'best_acc': best_acc
            }
            torch.save(checkpoint,
                       os.path.join(checkpoint_dir, "epoch_{}_val_acc_{:.3f}.pth".format(epoch, best_acc)))
            # torch.save(model, "{}/ckpt.model".format(model_dir))
            print('saving model with acc {:.3f}'.format(best_acc))

        print('Time: {:.3f}'.format(time.time() - epoch_start_time))
