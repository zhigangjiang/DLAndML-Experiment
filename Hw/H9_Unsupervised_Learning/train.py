import os
import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import numpy as np


def train(start_epoch, n_epoch, model, loader, loss, optimizer, best_loss, checkpoint_dir, device):
    # 主要的訓練過程
    for epoch in range(start_epoch, n_epoch):
        epoch_loss = 0
        for data in loader:
            img = data
            img = img.to(device)

            output1, output = model(img)
            train_loss = loss(output, img)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()

        cur_loss = epoch_loss / len(loader)
        if cur_loss < best_loss:
            best_loss = cur_loss
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'epoch_loss': best_loss
            }
            torch.save(checkpoint,
                       os.path.join(checkpoint_dir, "epoch_{}_train_loss_{:.3f}.pth".format(epoch, best_loss)))
            # torch.save(model, "{}/ckpt.model".format(model_dir))

            print('saving model with loss {:.3f}'.format(best_loss))

        print('epoch [{}/{}], loss:{:.5f}'.format(epoch + 1, n_epoch, cur_loss))


def inference(loader, model, device):
    latents = []
    for i, x in enumerate(loader):
        x = torch.FloatTensor(x)
        vec, img = model(x.to(device))
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            # middle code as latent
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis=0)
    print('Latents Shape:', latents.shape)
    return latents


def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded
