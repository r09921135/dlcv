"""
Code modified from : https://github.com/clvrai/ACGAN-PyTorch
"""
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import _netG


parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=110,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--log_path', default='./logs',
                    help='folder to example images')
parser.add_argument('--out_path', default='./output',
                    help='folder to output testing results')
parser.add_argument(
    '--model_path', default='./models/netG_0.903.pth', help='folder to models')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10,
                    help='Number of classes for AC-GAN')
parser.add_argument('--show_acc', action='store_true')

opt = parser.parse_args()


# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
opt.manualSeed = 9248
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('Device used:', device)

# some hyper parameters
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# Define the generator and initialize the weights
netG = _netG(nz)
netG.load_state_dict(torch.load(
    opt.model_path, map_location=torch.device(device)))
if device == 'cuda':
    netG.cuda()
netG.eval()

if opt.show_acc == True:
    from digit_classifier import *
    # Define the classifier for test the generation quality
    net = Classifier()
    path = "Classifier.pth"
    load_checkpoint(path, net, device)
    if device == 'cuda':
        net.cuda()
    net.eval()

tp = 0
imgs = torch.FloatTensor()
with torch.no_grad():
    for i in range(10):
        batch_size = 100

        noise_eval = torch.FloatTensor(batch_size, nz, 1, 1)
        if device == 'cuda':
            noise_eval = noise_eval.cuda()
        noise_eval = Variable(noise_eval)

        noise_eval.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), i] = 1
        noise_[np.arange(batch_size),
               :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise_eval.data.copy_(noise_.view(batch_size, nz, 1, 1))

        # Generate 100 images
        imgs_sample = (netG(noise_eval).data + 1) / 2.0
        imgs_sample = transforms.Resize(28)(imgs_sample)
        imgs = torch.cat((imgs, imgs_sample[:10].cpu()), 0)

        if opt.show_acc == True:
            # Evaluate generated images
            pred = net(imgs_sample)
            pred = torch.argmax(pred, dim=1)
            tp += (pred == i).sum()

        # Save the generated images
        for j in range(100):
            img_name = os.path.join(opt.out_path, f'{i}_{j+1:03d}.png')
            img_sample = transforms.ToPILImage()(imgs_sample[j].cpu())
            img_sample.save(img_name, format='png')

if opt.show_acc == True:
    print('Acc:', (tp/1000).item())

# filename = os.path.join(opt.log_path, 'examples.png')
# vutils.save_image(imgs, filename, nrow=10)

print('Done!')
