import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim 
import argparse
from dataset import *
from model import Model

def train(args):

    train_tfm = transforms.Compose([
                                # transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                # transforms.RandomHorizontalFlip(p=0.5),
                                # transforms.RandomRotation(35),
                                transforms.Resize((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)
                                ])

    valid_tfm = transforms.Compose([
                                transforms.Resize((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)
                                ])

    # load the training set
    train_set = Data(args.train_dir, transform=train_tfm)
    print('# images in train_set:', len(train_set)) # Should print 3680
    # load the validation set
    valid_set = Data(args.val_dir, transform=valid_tfm)
    print('# images in valid_set:', len(valid_set)) # Should print 1500

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    seed = 25
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cuda' if use_cuda else 'cpu'
    print('Device used:', device)

    # load model
    model = Model(pretrained=True).to(device)
    # for param in model.parameters():   
    #     param.requires_grad = False
    # fc_param = model.backbone.fc.parameters()
    # for param in fc_param:
    #     param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)
    
    print('Start training!')
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tp = 0
        train_loss = []
        model.train()
        for idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=-1)
            tp += (pred == labels).float().sum()
            train_loss.append(loss.item())

        scheduler.step()
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = tp / len(train_set)
        print('Train | Epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(epoch, train_loss, train_acc))

        if epoch % 1 == 0:
            with torch.no_grad():
                valid_acc = []
                valid_loss = []
                model.eval()
                for idx, (data, labels) in enumerate(valid_loader):
                    data, labels = data.to(device), labels.to(device)

                    output = model(data)
                    loss = criterion(output, labels)

                    acc = (output.argmax(dim=-1) == labels).float().mean()
                    valid_loss.append(loss.item())
                    valid_acc.append(acc)
                
                valid_loss = sum(valid_loss) / len(valid_loss)
                valid_acc = sum(valid_acc) / len(valid_acc)

                # save model
                if valid_acc > best_acc and valid_acc > 0.94:
                    best_acc = valid_acc
                    checkpoint = os.path.join(args.save_dir, '{:.4f}.pth.tar'.format(valid_acc))
                    torch.save(model.state_dict(), checkpoint)
                    print('Saving model...')
                    
                print('Valid | Epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(epoch, valid_loss, valid_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--train_dir', default='/home/rayting/Henry/DLCV/hw3/hw3_data/p1_data/train/', 
                    help='training images directory', type=str)
    parser.add_argument('--val_dir', default='/home/rayting/Henry/DLCV/hw3/hw3_data/p1_data/val/', 
                    help='validation images directory', type=str)
    parser.add_argument('--save_dir', default='.', 
                    help='save model directory', type=str)
    args = parser.parse_args()

    train(args)