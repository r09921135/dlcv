import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim 
from torchvision import models
import argparse
from dataset import *
from byol_pytorch import BYOL

def train(args):

    train_tfm = transforms.Compose([
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])

    # load the training set
    train_set = Data_mini(args.train_dir, transform=train_tfm)
    print('# images in train_set:', len(train_set))
    
    # create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    seed = 25
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cuda' if use_cuda else 'cpu'
    print('Device used:', device)

    # load model
    resnet = models.resnet50(pretrained=False)

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    )
    learner.to(device)
    
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print('Start training!')
    min_loss = 100
    for epoch in range(1, args.epochs + 1):
        train_loss = []
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            loss = learner(data)
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

            train_loss.append(loss.item())

        # scheduler.step()
        train_loss = sum(train_loss) / len(train_loss)
        print('Train | Epoch: {}, Loss: {:.5f}'.format(epoch, train_loss))

        # save lastest model
        checkpoint = os.path.join(args.save_dir, 'pretrained.pth')
        torch.save(resnet.state_dict(), checkpoint)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--train_dir', default='/home/rayting/Henry/DLCV/hw4/hw4_data/mini/train/', 
                    help='training images directory', type=str)
    parser.add_argument('--save_dir', default='.', 
                    help='save model directory', type=str)
    args = parser.parse_args()

    train(args)