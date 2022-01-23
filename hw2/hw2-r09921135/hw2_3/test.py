import os
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
import pandas as pd
import argparse

from data_loader import Data, Data_inf


def test(dataset_name, epoch, tsne=False):

    root_path = '/home/rayting/Henry/DLCV/hw2/hw2_data/digits/'
    model_root = os.path.join('.', 'models')
    image_root = os.path.join(root_path, dataset_name)

    cuda = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""
    if dataset_name == 'usps':
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    dataset = Data(
        image_root,
        'test',
        img_transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    """ inferencing """
    my_net = torch.load(os.path.join(
        model_root, 'model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    with torch.no_grad():
        while i < len_dataloader:

            # test model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            batch_size = len(t_label)

            input_img = torch.FloatTensor(
                batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)

            if cuda:
                t_img = t_img.cuda()
                t_label = t_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()

            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)

            class_output, _ = my_net(input_data=input_img, alpha=alpha)

            pred = torch.argmax(class_output, dim=1)
            n_correct += (pred == class_label).cpu().sum()
            n_total += batch_size

            i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    print('epoch: %d, accuracy of the %s dataset: %f' %
          (epoch, dataset_name, accu))


def test_inf(args):

    cuda = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""
    if args.target == 'usps':
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    # load target domain data
    dataset = Data_inf(args.data_path, img_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    """ evaluating """
    my_net = torch.load(args.model_path)
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    pred_total = torch.FloatTensor()
    names = []

    with torch.no_grad():
        while i < len_dataloader:

            # test model using target data
            data_target = data_target_iter.next()
            t_img, t_name = data_target

            batch_size = len(t_img)

            input_img = torch.FloatTensor(
                batch_size, 3, image_size, image_size)
            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
            input_img.resize_as_(t_img).copy_(t_img)

            class_output, _ = my_net(input_data=input_img, alpha=alpha)
            pred = torch.argmax(class_output, dim=1)
            pred_total = torch.cat((pred_total, pred.float().cpu()), 0)
            names.extend(t_name)
            i += 1

    pred_total = pred_total.numpy()
    with open((args.out_path), 'w') as f:
        f.write('image_name,label\n')
        for i, y in enumerate(pred_total):
            f.write('{},{}\n'.format(names[i], int(y)))

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', default='/home/rayting/Henry/DLCV/hw2/hw2_data/digits/usps/test')
    parser.add_argument('--target', default='usps', help='target domain name')
    parser.add_argument('--out_path', default='./pred.csv',
                        help='output csv results')
    parser.add_argument('--model_path', default='./models/model_0.471.pth')
    parser.add_argument('--bonus', action='store_true',
                        help='run the imporved model')
    args = parser.parse_args()

    if args.bonus == True:
        if args.target == 'mnistm':
            args.model_path = './hw2_3/model_0.610.pth'
        elif args.target == 'usps':
            args.model_path = './hw2_3/model_0.879.pth'
        else:
            args.model_path = './hw2_3/model_0.297.pth'
    else:
        if args.target == 'mnistm':
            args.model_path = './hw2_3/model_0.471.pth'
        elif args.target == 'usps':
            args.model_path = './hw2_3/model_0.794.pth'
        else:
            args.model_path = './hw2_3/model_0.280.pth'

    test_inf(args)
