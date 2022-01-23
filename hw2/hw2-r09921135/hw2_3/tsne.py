import os
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
from sklearn import manifold
import pandas as pd
import numpy as np

from data_loader import Data


def tsne_plot(args):

    source_image_root = os.path.join(args.data_root, args.source)
    target_image_root = os.path.join(args.data_root, args.target)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    # load data
    if args.source == 'usps':
        img_transform_source = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    else:
        img_transform_source = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    if args.target == 'usps':
        img_transform_target = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    else:
        img_transform_target = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    # load source domain data
    dataset_source = Data(
        source_image_root,
        'test',
        img_transform_source
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=False,
    )

    # load target domain data
    dataset_target = Data(
        target_image_root,
        'test',
        img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=False,
    )

    """ inferencing """
    my_net = torch.load(args.model_path)
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    # feat_total = torch.FloatTensor().cuda()
    labels_total = torch.FloatTensor().cuda()
    domain_total = torch.FloatTensor()
    X_TSNE = []

    dataset_name = [args.source, args.target]
    for d, dataloader in enumerate([dataloader_source, dataloader_target]):
        len_dataloader = len(dataloader)
        data_target_iter = iter(dataloader)

        i = 0
        n_total = 0
        n_correct = 0
        feat_total = torch.FloatTensor().cuda()
        with torch.no_grad():
            while i < len_dataloader:

                # test model using target data
                data_target = data_target_iter.next()
                t_img, t_label = data_target

                batch_size = len(t_label)

                input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
                class_label = torch.LongTensor(batch_size)

                if cuda:
                    t_img = t_img.cuda()
                    t_label = t_label.cuda()
                    input_img = input_img.cuda()
                    class_label = class_label.cuda()

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)

                class_output, _ = my_net(input_data=input_img, alpha=alpha)

                # store features
                feat = my_net.feature(input_img.expand(input_img.data.shape[0], 3, 28, 28))
                feat = feat.view(-1, 50 * 4 * 4)
                feat_total = torch.cat((feat_total, feat), 0)
                labels_total = torch.cat((labels_total, class_label.float()), 0)
                if d == 0:
                    domain = torch.zeros((batch_size))
                else:
                    domain = torch.ones((batch_size))
                domain_total = torch.cat((domain_total,domain), 0)

                pred = torch.argmax(class_output, dim=1)
                n_correct += (pred == class_label).cpu().sum()
                n_total += batch_size
                i += 1

        accu = n_correct.data.numpy() * 1.0 / n_total
        print('accuracy of the %s dataset: %f' % (dataset_name[d], accu))

        feat = feat_total.cpu().numpy()

        # perform tsne
        X_tsne = manifold.TSNE(early_exaggeration=50, random_state=250).fit_transform(feat)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
        X_TSNE.append(X_norm)
    
    labels = labels_total.cpu().numpy()
    domains = domain_total.cpu().numpy()
    X_norm = np.concatenate((X_TSNE[0], X_TSNE[1]), 0)
    
    # plot tsne results
    df = pd.DataFrame(dict(Feature_1=X_norm[:,0], Feature_2=X_norm[:,1], category=labels))
    fig = df.plot(x="Feature_1", y="Feature_2", kind='scatter', s=1, c='category', colormap='viridis')
    fig.figure.savefig(os.path.join(args.output_path, 'tsne_result_1.png'))

    df = pd.DataFrame(dict(Feature_1=X_norm[:,0], Feature_2=X_norm[:,1], domain=domains))
    fig = df.plot(x="Feature_1", y="Feature_2", kind='scatter', s=1, c='domain', colormap='viridis')
    fig.figure.savefig(os.path.join(args.output_path, 'tsne_result_2.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/rayting/Henry/DLCV/hw2/hw2_data/digits/')
    parser.add_argument('--source', default='usps', help='source domain name')
    parser.add_argument('--target', default='svhn', help='target domain name')
    parser.add_argument('--output_path', default='./output', help='output tsne results')
    parser.add_argument('--model_path', default='./models/model_0.280.pth')    
    args = parser.parse_args()

    tsne_plot(args)
