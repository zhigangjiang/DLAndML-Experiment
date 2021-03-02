import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import os
import matplotlib.pyplot as plt


def test(n_output, checkpoint_dir, G, z_dim, device):
    # generate images and save the result
    z_sample = Variable(torch.randn(n_output, z_dim)).to(device)
    imgs_sample = (G(z_sample).data + 1) / 2.0
    save_dir = os.path.join(checkpoint_dir, 'logs')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    filename = os.path.join(save_dir, f'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    # show image
    grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
