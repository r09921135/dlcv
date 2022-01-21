import torch
import random
import numpy as np
import os
import sys
import glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
import pandas as pd


class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            # Actually we need to upscale to next full batch
            return sum([len(bucket) for bucket in self.buckets])
        else:
            # Ensures every instance has the chance to be visited in an epoch
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num


class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label


class test_Dataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.fnames = glob.glob(os.path.join(root, '*'))
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname).convert('RGB')
        img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return self.num_samples

class RoTTrans():
    def __init__(self, base_trans):
        self.base_trans = base_trans
        self.rotate_0 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.rotate_90 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.rotate(90, expand=True)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.rotate_180 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.rotate(180, expand=True)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.rotate_270 =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.rotate(270, expand=True)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        y1 = self.base_trans(img)
        label = np.random.randint(4, size=1)
        if label[0] == 0:
            y2 = self.rotate_0(img)
        elif label[0] == 1:
            y2 = self.rotate_90(img)
        elif label[0] == 2:
            y2 = self.rotate_180(img)
        elif label[0] == 3:
            y2 = self.rotate_270(img)

        return y1, y2, label[0]

class FoodLTDataLoader(DataLoader):
    """
    ImageNetLT Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True):
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
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if training:
            # dataset = LT_Dataset(data_dir,  train_txt, train_trsfm)
            # val_dataset = LT_Dataset(data_dir, val_txt, test_trsfm)
            dataset = datasets.DatasetFolder(os.path.join(data_dir, 'train'), loader=lambda x: Image.open(
                x), extensions="jpg", transform=RoTTrans(train_trsfm))
            val_dataset = datasets.DatasetFolder(
                os.path.join(data_dir, 'val'), loader=lambda x: Image.open(x), extensions="jpg", transform=test_trsfm)
        else:  # test
            # dataset = LT_Dataset(data_dir, test_txt, test_trsfm)
            dataset = test_Dataset(data_dir, test_trsfm)
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 1000

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print(
                    "Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        # Note that sampler does not apply to validation set
        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler)

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)
