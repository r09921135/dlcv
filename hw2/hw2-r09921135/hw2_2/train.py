"""
Code modified from : https://github.com/clvrai/ACGAN-PyTorch
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from qqdm import qqdm
from utils import weights_init, compute_acc
from model import _netD, _netG
from dataset import Data
from digit_classifier import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--resf', default='.', help='folder to output testing results')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')

opt = parser.parse_args()


try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
opt.manualSeed = 9248
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# prepare dataset
# data_root = '../hw2_data/digits/mnistm/'
# dataset = Data(
#     root = opt.dataroot, 
#     transform=transforms.Compose([
#         transforms.Resize(opt.imageSize),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]))
# torch.save(dataset, 'dataset.pt')
dataset = torch.load('dataset.pt')

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers),
                                         drop_last=True)

# some hyper parameters
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# Define the generator and initialize the weights
netG = _netG(nz)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

# Define the discriminator and initialize the weights
netD = _netD(num_classes)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

# Define the classifier for test the generation quality
net = Classifier()
path = "Classifier.pth"
load_checkpoint(path, net)

# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(opt.batchSize)
aux_label = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    net.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
eval_label = np.random.randint(0, num_classes, opt.batchSize)
eval_onehot = np.zeros((opt.batchSize, num_classes))
eval_onehot[np.arange(opt.batchSize), eval_label] = 1
eval_noise_[np.arange(opt.batchSize), :num_classes] = eval_onehot[np.arange(opt.batchSize)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0

for epoch in range(opt.niter):
    netD.train()
    netG.train()
    progress_bar = qqdm(dataloader)
    for i, data in enumerate(progress_bar, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.data.resize_as_(real_cpu).copy_(real_cpu)
        # soft label
        dis_label.data.resize_(batch_size).fill_(real_label) + (torch.rand(batch_size)*0.2-0.1).cuda()
        aux_label.data.resize_(batch_size).copy_(label)
        dis_output, aux_output = netD(input)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        dis_label.data.fill_(fake_label) + (torch.rand(batch_size)*0.1).cuda()
        # noisy input dor discriminator
        dis_output, aux_output = netD((fake.detach()) + (torch.rand(batch_size, 3, opt.imageSize, opt.imageSize)*0.1-0.05).cuda())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label) # + (torch.rand(batch_size)*0.2-0.1).cuda()  # fake labels are real for generator cost
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        errG = dis_errG + aux_errG
        errG.backward()
        D_G_z2 = dis_output.data.mean()
        optimizerG.step()

        # compute the average loss
        curr_iter = epoch * len(dataloader) + i
        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter
        all_loss_A = avg_loss_A * curr_iter
        all_loss_G += errG.item()
        all_loss_D += errD.item()
        all_loss_A += accuracy
        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)
        avg_loss_A = all_loss_A / (curr_iter + 1)

        progress_bar.set_infos({
            'Loss_D': round(avg_loss_D, 3),
            'Loss_G': round(avg_loss_G, 3),
            'D(x)': round(D_G_z1.item(), 3),
            'D(G(z))': round(D_G_z2.item(), 3),
            'Acc': round(avg_loss_A, 3),
            'Epoch': epoch + 1,
        })
        
        if i % 100 == 0:
            fake = netG(eval_noise)
            vutils.save_image(
                (fake.data + 1) / 2.0,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
            )


    if (epoch+1) >= 15:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))

        # generate 100 images for 10 classes
        # netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (opt.outf, epoch)))
        netG.eval()
        tp = 0
        imgs = torch.FloatTensor().cuda()
        for i in range(10):
            batch_size = 100
            
            noise_eval = torch.FloatTensor(batch_size, nz, 1, 1).cuda()
            noise_eval = Variable(noise_eval)

            noise_eval.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            label = np.random.randint(0, num_classes, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            class_onehot = np.zeros((batch_size, num_classes))
            class_onehot[np.arange(batch_size), i] = 1
            noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
            noise_ = (torch.from_numpy(noise_))
            noise_eval.data.copy_(noise_.view(batch_size, nz, 1, 1))

            # Generate 100 images
            imgs_sample = (netG(noise_eval).data + 1) / 2.0
            imgs_sample = transforms.Resize(28)(imgs_sample)
            imgs = torch.cat((imgs, imgs_sample[:10]), 0)

            # test generated images quality
            pred = net(imgs_sample)
            pred = torch.argmax(pred, dim=1)
            tp += (pred==i).sum()

            # Save the generated images
            os.makedirs(opt.resf, exist_ok=True)
            for j in range(100):
                img_name = os.path.join(opt.resf, f'{i}_{j+1:03d}.png')
                img_sample = transforms.ToPILImage()(imgs_sample[j].cpu())
                img_sample.save(img_name, format='png')

        print('Acc:', (tp/1000).item())