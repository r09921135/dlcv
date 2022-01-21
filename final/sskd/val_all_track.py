import argparse
import os
import torch
from tqdm import tqdm
from pathlib import Path
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F
import torch
import random
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
from PIL import ImageFilter
import glob
import pandas as pd
from utils import load_state_dict, rename_parallel_state_dict


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class test_Dataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.fnames = glob.glob(os.path.join(root, '*'))
        self.fnames.sort()
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname).convert('RGB')
        img = self.transform(img)
        label = 0
        id = fname.split('/')[-1].split('.')[0]
        return img, label, id

    def __len__(self):
        return self.num_samples


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    train_dir = dir.split('val')[0] + 'train'
    # print(train_dir)
    data_num_in_class = {classes[i]: len(os.listdir(
        os.path.join(train_dir, classes[i]))) for i in range(len(classes))}
    return classes, class_to_idx, data_num_in_class


def make_dataset(dir, class_to_idx, data_num_in_class):
    datas = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        num = data_num_in_class[target]
        if num >= 100:
            track = 'f'
        elif num < 10:
            track = 'r'
        else:
            track = 'c'

        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target], track)
                datas.append(item)
    return datas


class val_Dataset(Dataset):
    def __init__(self, root, transform, find_classes, make_dataset):
        self.transform = transform
        self.root = root
        self.find_classes = find_classes
        self.classes, self.class_to_idx, self.data_num_in_class = self.find_classes(
            self.root)
        self.make_dataset = make_dataset
        self.datas = self.make_dataset(
            self.root, self.class_to_idx, self.data_num_in_class)

    def __getitem__(self, idx):
        fname, label, track = self.datas[idx]
        img = Image.open(fname).convert('RGB')
        img = self.transform(img)
        return img, label, track

    def __len__(self):
        return len(self.datas)


class FoodLTDataLoader(DataLoader):
    """
    ImageNetLT Data Loader
    """

    def __init__(self, data_dir, batch_size, num_workers=1, training=True):
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop((224, 224)),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
        ])

        if training:
            dataset = datasets.DatasetFolder(os.path.join(data_dir, 'train'), loader=lambda x: Image.open(
                x), extensions="jpg", transform=train_trsfm)
            train_dataset = datasets.DatasetFolder(os.path.join(data_dir, 'train'), loader=lambda x: Image.open(
                x), extensions="jpg", transform=train_trsfm)
            val_dataset = datasets.DatasetFolder(
                os.path.join(data_dir, 'val'), loader=lambda x: Image.open(x), extensions="jpg", transform=test_trsfm)
        else:  # test
            # print(os.path.join(data_dir, 'val'))
            dataset = val_Dataset(os.path.join(
                data_dir, 'val'), test_trsfm, find_classes, make_dataset)
            train_dataset = val_Dataset(
                os.path.join(data_dir, 'val'), TwoCropsTransform(train_trsfm), find_classes, make_dataset)
            val_dataset = val_Dataset(
                os.path.join(data_dir, 'val'), test_trsfm, find_classes, make_dataset)

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)
        # self.val = datasets.DatasetFolder(
        #     os.path.join(os.path.join(data_dir, 'val')), loader=lambda x: Image.open(x), extensions="jpg", transform=test_trsfm)
        # self.mapping = self.val.class_to_idx

        self.shuffle = False
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        # Note that sampler does not apply to validation set
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def train_set(self):
        return DataLoader(dataset=self.train_dataset, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)


def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def resume_checkpoint(resume_path, model, state_dict_only=True):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)

    state_dict = checkpoint['state_dict']
    if state_dict_only:
        rename_parallel_state_dict(state_dict)

    # self.model.load_state_dict(state_dict)
    load_state_dict(model, state_dict)


def main(config):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    print(config.resume)
    state_dict = checkpoint['state_dict']
    rename_parallel_state_dict(state_dict)
    # model.load_state_dict(state_dict)
    load_state_dict(model, state_dict)

    # prepare model for testing
    model = model.to(device)
    data_loader = FoodLTDataLoader(
        config['data_loader']['args']['data_dir'],
        batch_size=32,
        training=False,
        num_workers=0,
    )
    valid_data_loader = data_loader.test_set()
    test_validation(valid_data_loader, model, device)


def test_validation(data_loader, model, device):
    model.eval()
    prediction_results = []
    labels = []
    tracks = []
    with torch.no_grad():
        for i, (data, label, track) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            b, crop, c, h, w = data.shape
            output = model(data.view(-1, c, h, w))
            expert1_logits_output = output['logits'][:, 0, :].view(
                b, crop, -1).mean(1)
            expert2_logits_output = output['logits'][:, 1, :].view(
                b, crop, -1).mean(1)
            expert3_logits_output = output['logits'][:, 2, :].view(
                b, crop, -1).mean(1)
            aggregation_output = (
                expert1_logits_output + expert2_logits_output + expert3_logits_output)/3
            prediction_results.extend(
                aggregation_output.argmax(axis=1).cpu().numpy().tolist())
            labels.extend(label)
            tracks.extend(track)
    freq_num = 0
    comm_num = 0
    rare_num = 0
    freq_hit = 0
    comm_hit = 0
    rare_hit = 0
    tot_hit = 0
    for i in range(len(prediction_results)):
        if tracks[i] == 'f':
            freq_num += 1
            if prediction_results[i] == labels[i]:
                freq_hit += 1
                tot_hit += 1
        elif tracks[i] == 'c':
            comm_num += 1
            if prediction_results[i] == labels[i]:
                comm_hit += 1
                tot_hit += 1
        elif tracks[i] == 'r':
            rare_num += 1
            if prediction_results[i] == labels[i]:
                rare_hit += 1
                tot_hit += 1
        else:
            print('error')
    print('Total accuracy: ', tot_hit/len(prediction_results))
    print('Freq. track accuracy: ', freq_hit/freq_num)
    print('Comm. track accuracy: ', comm_hit/comm_num)
    print('Rare. track accuracy: ', rare_hit/rare_num)
    print('Total hit, total_num: ', tot_hit, '   ', len(prediction_results))
    print('Freq hit, Freq_num: ', freq_hit, '   ', freq_num)
    print('Comm hit, Comm_num: ', comm_hit, '   ', comm_num)
    print('Rare hit, Rare_num: ', rare_hit, '   ', rare_num)
    print('finish!')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--epochs', default=1, type=int,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    main(config)
