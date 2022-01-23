import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
import argparse
from dataset import *
from model import Model
from sklearn import manifold
import pandas as pd

def test(args):
    # mean = [0.5071, 0.4809, 0.4305]
    # std = [0.2628, 0.2547, 0.2738]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_tfm = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    # load the validation set
    if args.with_acc:
        test_set = Data(args.test_dir, transform=test_tfm)
    else:
        test_set = Data_inf(args.test_dir, transform=test_tfm)
    print('# images in test_set:', len(test_set)) # Should print 257

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
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint)

    # inference
    print('Start inferencing!')
    model.eval()
    if args.with_acc:
        with torch.no_grad():
            valid_acc = []
            pred_total = []
            feat_total = torch.FloatTensor().to(device)
            labels_total = torch.FloatTensor()
            for idx, (data, labels) in enumerate(test_loader):
                data = data.to(device)

                output = model(data)

                if args.tsne:
                    feat = model.backbone(data)
                    feat = feat.reshape(-1, feat.shape[1])
                    feat_total = torch.cat((feat_total, feat), 0)
                    labels_total = torch.cat((labels_total, labels.float()), 0)

                pred = output.argmax(dim=-1).cpu()
                pred_total.append(pred.numpy())

                acc = (pred == labels).float().mean()
                valid_acc.append(acc)
            
            valid_acc = sum(valid_acc) / len(valid_acc)
            print('Test acc: {:.5f}'.format(valid_acc))

        # perform tsne
        if args.tsne:
            feat = feat_total.cpu().numpy()
            labels = labels_total.cpu().numpy()

            X_tsne = manifold.TSNE(init='pca', early_exaggeration=20, random_state=5).fit_transform(feat)

            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize

            df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=labels))
            fig = df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')
            fig.figure.savefig('tsne_result.png')
        
    else:
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
            f.write('image_id,label\n')
            for i, y in enumerate(pred_total):
                f.write('{},{}\n'.format(names[i], int(y)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./hw1_1/0.8256.pth.tar', 
                    help='save model directory', type=str)
    parser.add_argument('--test_dir', default='/home/rayting/Henry/DLCV/hw1/hw1_data/p1_data/val_50/', 
                    help='test images directory', type=str)
    parser.add_argument('--out_dir', default='./pred.csv', 
                    help='output images directory', type=str)
    parser.add_argument('--with_acc', action='store_true')               
    parser.add_argument('--tsne', action='store_true')
    args = parser.parse_args()

    test(args)