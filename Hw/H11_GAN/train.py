import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import os
import matplotlib.pyplot as plt


def train(n_epoch, checkpoint_dir, dataloader, G, optimizer_G, D, optimizer_D, loss,  z_dim, device):
    # for logging
    z_sample = Variable(torch.randn(100, z_dim)).to(device)

    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim)).to(device)
            r_imgs = Variable(imgs).to(device)
            f_imgs = G(z)

            # label
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()

            # dis
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # compute loss
            r_loss = loss(r_logit, r_label)
            f_loss = loss(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # update model
            D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim)).to(device)
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)

            # compute loss
            loss_G = loss(f_logit, r_label)

            # update model
            G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # log
            print(
                f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
                end='')
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(checkpoint_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        # show generated image
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()
        if (e + 1) % 5 == 0:
            torch.save(G.state_dict(), os.path.join(checkpoint_dir, f'dcgan_g.pth'))
            torch.save(D.state_dict(), os.path.join(checkpoint_dir, f'dcgan_d.pth'))
