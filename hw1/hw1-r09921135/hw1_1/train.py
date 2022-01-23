import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
import argparse
from dataset import *
from model import Model

def train(args):
    # calculate the mean and std of the dataset
    # t_set = Data(args.train_dir, transform=transforms.ToTensor())
    # v_set = Data(args.val_dir, transform=transforms.ToTensor())
    # concat_set = ConcatDataset([t_set, v_set])
    # loader = DataLoader(concat_set, batch_size=1, shuffle=False)
    # mean, std = cal_mean_and_std(loader)
    # print('Data mean:', mean, 'Data std:', std)
    # mean = [0.5071, 0.4809, 0.4305]
    # std = [0.2628, 0.2547, 0.2738]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
                                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(35),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])

    valid_tfm = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])

    # load the training set
    train_set = Data(args.train_dir, transform=train_tfm)
    # load the validation set
    valid_set = Data(args.val_dir, transform=valid_tfm)
    print('# images in train_set:', len(train_set)) # Should print 22500
    print('# images in valid_set:', len(valid_set)) # Should print 2500

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,14], gamma=0.1)
    
    print('Start training!')
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_acc = []
        train_loss = []
        model.train()
        for idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_acc.append(acc)

        scheduler.step()
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)
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
                if valid_acc > best_acc and valid_acc > 0.8:
                    best_acc = valid_acc
                    checkpoint = os.path.join(args.save_dir, '{:.4f}.pth.tar'.format(valid_acc))
                    torch.save(model.state_dict(), checkpoint)
                    print('Saving model...')
                    
                print('Valid | Epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(epoch, valid_loss, valid_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--weight_decay', default=2e-4, type=float)
    parser.add_argument('--train_dir', default='/home/rayting/Henry/DLCV/hw1/hw1_data/p1_data/train_50/', 
                    help='training images directory', type=str)
    parser.add_argument('--val_dir', default='/home/rayting/Henry/DLCV/hw1/hw1_data/p1_data/val_50/', 
                    help='validation images directory', type=str)
    parser.add_argument('--save_dir', default='.', 
                    help='save model directory', type=str)
    args = parser.parse_args()

    train(args)