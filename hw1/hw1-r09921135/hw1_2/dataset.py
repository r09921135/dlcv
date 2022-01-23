import numpy as np
import imageio
import PIL
import os
import torch
import torchvision.transforms as transforms 
from mean_iou_evaluate import read_masks
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, path, type, transform=None):
        """ Intialize the dataset """
        self.type = type
        self.filenames = []

        # read images
        self.filenames = [file for file in os.listdir(path) if file.endswith('.jpg')]
        self.filenames.sort()
        self.len = len(self.filenames)
        
        images = torch.FloatTensor()
        for fn in self.filenames:
            image = PIL.Image.open(os.path.join(path, fn))
            if transform is not None:
                image = transform(image)
            images = torch.cat((images, image.unsqueeze(0)), 0)
        self.images = images
        # np.save(type + '_img.npy', images.numpy())
        # self.images = np.load(type + '_img.npy')

        # read masks
        self.masks = read_masks(path).astype(np.float32)
        # np.save(type + '_mask.npy', self.masks)
        # self.masks = np.load(type + '_mask.npy')

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image = self.images[index]
        mask = self.masks[index]
        mask = mask[::2,::2] if self.type == 'train' else mask
        name = (self.filenames[index]).split('.jpg')[0]

        return image, mask, name

    def __len__(self):
        return self.len


class Data_inf(Dataset):
    def __init__(self, path, transform=None):
        """ Intialize the dataset """
        self.filenames = []

        # read images
        self.filenames = [file for file in os.listdir(path) if file.endswith('.jpg')]
        self.filenames.sort()
        self.len = len(self.filenames)
        
        images = torch.FloatTensor()
        for fn in self.filenames:
            image = PIL.Image.open(os.path.join(path, fn))
            if transform is not None:
                image = transform(image)
            images = torch.cat((images, image.unsqueeze(0)), 0)
        self.images = images
        # np.save(type + '_img.npy', images.numpy())
        # self.images = np.load(type + '_img.npy')

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image = self.images[index]
        name = (self.filenames[index]).split('.jpg')[0]

        return image, name

    def __len__(self):
        return self.len


def cal_mean_and_std(loader):
    """Compute the mean and std of the dataset
    
        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data, _ in loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

        
if __name__ == '__main__':
    train_data_path = '/home/rayting/Henry/DLCV/hw1/hw1_data/p2_data/train/'
    valid_data_path = '/home/rayting/Henry/DLCV/hw1/hw1_data/p2_data/validation/'
    
    train_tfm = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),])

    # load the training set
    train_set = Data(train_data_path, 'train', transform=train_tfm)
    print('train image shape:', train_set[0][0].shape, 'train mask shape:', train_set[0][1].shape)

    # load the validation set
    valid_set = Data(valid_data_path, 'valid', transform=train_tfm)
    print('valid image shape:', valid_set[0][0].shape, 'valid mask shape:', valid_set[0][1].shape)

    # load the testing set
    test_set = Data_inf(valid_data_path, transform=train_tfm)
    print('testing sample:', test_set[1][0].shape, ',', test_set[1][1])
