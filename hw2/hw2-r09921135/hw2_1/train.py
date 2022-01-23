import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from qqdm import qqdm
import random

from dataset import Data
from model import *


# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
seed = random.randint(1, 10000)
print("Random Seed: ", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cuda' if use_cuda else 'cpu'
print('Device used:', device)


train_tfm = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# load the training set
train_data_path = './hw2_data/face/train/'
train_set = Data(train_data_path, transform=train_tfm)
print('# images in train_set:', len(train_set)) # Should print 40000

# show images
# images = [(train_set[i] + 1) / 2 for i in range(16)]
# grid_img = torchvision.utils.make_grid(images, nrow=4)
# plt.figure(figsize=(10,10))
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()

# create dataloader
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Training hyperparameters
z_dim = 100
z_sample = torch.randn(100, z_dim, 1, 1).to(device)
lr = 2e-4
n_epoch = 100

log_dir = os.path.join('.', 'logs')
ckpt_dir = os.path.join('.', 'models')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
G = Generator().to(device)
D = Discriminator().to(device)
G.train()
D.train()
G.apply(weights_init)
D.apply(weights_init)

# Loss
criterion = nn.BCELoss()

# Optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

steps = 0
for e, epoch in enumerate(range(n_epoch)):
    progress_bar = qqdm(train_loader)
    for i, data in enumerate(progress_bar):
        imgs = data

        bs = imgs.size(0)

        # ============================================
        #  Train D
        # ============================================
        z = torch.randn(bs, z_dim, 1, 1).to(device)
        r_imgs = imgs.to(device)
        f_imgs = G(z)

        # Label
        r_label_noise = (torch.rand(bs)*0.1)
        f_label_noise = (torch.rand(bs)*0.1)
        r_label = torch.ones((bs)).to(device)
        f_label = torch.zeros((bs)).to(device)

        # Model forwarding
        r_logit = D(r_imgs.detach() + (r_label_noise.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 3, 64, 64).to(device)*0.5)).view(-1)
        f_logit_D = D(f_imgs.detach()).view(-1)
        
        # Compute the loss for the discriminator.
        r_loss = criterion(r_logit, r_label - r_label_noise.to(device))
        f_loss = criterion(f_logit_D, f_label + f_label_noise.to(device))
        loss_D = (r_loss + f_loss) / 2

        # Model backwarding
        D.zero_grad()
        loss_D.backward()

        # Update the discriminator.
        opt_D.step()

        # ============================================
        #  Train G
        # ============================================
        # Model forwarding
        f_logit_G = D(f_imgs).view(-1)
        
        # Compute the loss for the generator.
        loss_G = criterion(f_logit_G, r_label)

        # Model backwarding
        G.zero_grad()
        loss_G.backward()

        # Update the generator.
        opt_G.step()

        steps += 1
        
        # Set the info of the progress bar
        progress_bar.set_infos({
            'Loss_D': round(loss_D.item(), 1),
            'Loss_G': round(loss_G.item(), 1),
            'f_logit_D': round(f_logit_D.mean().item(), 4),
            'f_logit_G': round(f_logit_G.mean().item(), 4),
            'Epoch': e+1,
        })

    # evaluation
    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')
    G.train()

    # save checkpoints
    if (epoch+1) >= 30:
        torch.save(G.state_dict(), os.path.join(ckpt_dir, f'G_{epoch+1}.pth'))
        # torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

'''
for epoch in range(35, 40):
    print(epoch+1)
    g = Generator()
    g.load_state_dict(torch.load(os.path.join(ckpt_dir, f'G_{epoch+1}.pth')))
    g.eval()
    g.cuda()

    # Generate 1000 images and make a grid to save them.
    n_output = 1000
    z_sample = torch.randn(n_output, z_dim, 1, 1).cuda()
    imgs_sample = (g(z_sample).data + 1) / 2.0
    # log_dir = os.path.join('.', 'logs')
    # filename = os.path.join(log_dir, 'result.jpg')
    # torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    # # Show 32 of the images.
    # grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
    # plt.figure(figsize=(10,10))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.show()

    # Save the generated images.
    os.makedirs('output', exist_ok=True)
    for i in range(1000):
        torchvision.utils.save_image(imgs_sample[i], f'output/{i+1}.jpg')

    os.system('python -m pytorch_fid ./hw2_data/face/test/ ./output --device cuda:0')
'''