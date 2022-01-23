import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
import argparse
from dataset import *
from model import Model


def test(args):
    test_tfm = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    # load the validation set
    test_set = Data_inf(args.test_dir, transform=test_tfm)
    print('# images in test_set:', len(test_set)) 

    # create dataloader
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    seed = 25
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cuda' if use_cuda else 'cpu'
    print('Device used:', device)

    # load model
    print('Loading model...')
    model = Model().to(device)
    model.eval()
    vit = model.backbone
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint)

    # evaluating
    print('Start evaluating!')
    with torch.no_grad():
        names = []
        pred_total = []
        feat_total = torch.FloatTensor().to(device)
        for idx, (data, name) in enumerate(test_loader):
            data = data.to(device)

            output = model(data)

            pred = output.argmax(dim=-1).cpu()
            pred_total.append(pred.numpy())
            names.append(name[0])
    
    with open((args.out_dir), 'w') as f:
        f.write('filename,label\n')
        for i, y in enumerate(pred_total):
            f.write('{},{}\n'.format(names[i], int(y)))
            
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./0.9500.pth.tar', 
                    help='save model directory', type=str)
    parser.add_argument('--test_dir', default='/home/rayting/Henry/DLCV/hw3/hw3_data/p1_data/val/', 
                    help='test images directory', type=str)
    parser.add_argument('--out_dir', default='./pred.csv', 
                    help='output images directory', type=str)       
    args = parser.parse_args()

    test(args)