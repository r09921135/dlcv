import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim 
import argparse
from dataset import *
from model import Model

def train(args):

    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    # load the training set
    train_set = Data_office(args.data_dir, 'train', transform=tfm)
    print('# images in train_set:', len(train_set)) 
    # load the validation set
    valid_set = Data_office(args.data_dir, 'val', transform=tfm)
    print('# images in valid_set:', len(valid_set))

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
    model_path = './pretrained.pth'
    model = Model(pretrained=model_path).to(device)
    # for param in model.backbone.parameters():   
    #     param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.5)
    
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
                tp = 0
                valid_loss = []
                model.eval()
                for idx, (data, labels) in enumerate(valid_loader):
                    data, labels = data.to(device), labels.to(device)

                    output = model(data)
                    loss = criterion(output, labels)

                    pred = output.argmax(dim=-1)
                    tp += (pred == labels).float().sum()
                    valid_loss.append(loss.item())
                
                valid_loss = sum(valid_loss) / len(valid_loss)
                valid_acc = tp / len(valid_set)

                # save model
                if valid_acc > best_acc and valid_acc > 0.36:
                    best_acc = valid_acc
                    checkpoint = os.path.join(args.save_dir, '{:.4f}.pth.tar'.format(valid_acc))
                    torch.save(model.state_dict(), checkpoint)
                    print('Saving model...')
                    
                print('Valid | Epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(epoch, valid_loss, valid_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--data_dir', default='/home/rayting/Henry/DLCV/hw4/hw4_data/office/', 
                    help='images directory', type=str)
    parser.add_argument('--save_dir', default='.', 
                    help='save model directory', type=str)
    args = parser.parse_args()

    train(args)