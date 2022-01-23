import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import cv2
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
    test_set = Data(args.test_dir, transform=test_tfm)
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
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint)


    if args.viz == True:
        # visulaize positional embeddings
        pos_embed = model.backbone.positional_embedding.pos_embedding

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        fig = plt.figure(figsize=(10, 10))
        plt.title('Visualization results', fontsize=50)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.xlabel('Input patch column', fontsize=20)
        plt.ylabel('Input patch row', fontsize=20)
        for i in range(1, pos_embed.shape[1]):
            sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
            sim = sim.reshape((24, 24)).detach().cpu().numpy()
            ax = fig.add_subplot(24, 24, i)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(sim)
        plt.savefig('pos_embed.png')

        # visulaize self-attention
        img_name = ['26_5064.jpg', '29_4718.jpg', '31_4838.jpg']
        for name in img_name:
            img = Image.open(os.path.join(args.test_dir, name))
            x = test_tfm(img).unsqueeze(0).to(device)

            _ = model(x)

            attn_map = model.backbone.transformer.blocks[-1].attn.scores
            attn_map = attn_map.mean(1)  # (B, S, S)

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(transforms.Resize((384, 384))(img))

            attn_map = attn_map[0, 0, 1:].reshape((24, 24)).detach().cpu().numpy()
            attn_map = cv2.resize(attn_map, (384, 384))
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(attn_map, cmap='jet')
            plt.savefig(f'{name}_attn.png')


    with torch.no_grad():
        tp = 0
        for idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)

            output = model(data)

            pred = output.argmax(dim=-1).cpu()
            tp += (pred == labels).float().sum()
        
        acc = tp / len(test_set)
        print('Test acc: {:.5f}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./0.9500.pth.tar', 
                    help='save model directory', type=str)
    parser.add_argument('--test_dir', default='/home/rayting/Henry/DLCV/hw3/hw3_data/p1_data/val/', 
                    help='test images directory', type=str)
    parser.add_argument('--viz', action='store_true')
    args = parser.parse_args()

    test(args)