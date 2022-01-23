import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image


class Data(Dataset):
    def __init__(self, path, transform=None):
        """ Intialize the dataset """
        self.path = path
        self.transform = transform
        self.filenames = []
        self.num_class = 50

        # read filenames
        for i in range(self.num_class):
            filenames = [filename for filename in os.listdir(self.path) if filename.startswith(str(i) + '_')]
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(os.path.join(self.path, image_fn))
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class Data_inf(Dataset):
    def __init__(self, path, transform=None):
        """ Intialize the dataset """
        self.path = path
        self.transform = transform
        self.filenames = []
        self.num_class = 50

        # read filenames
        self.filenames = [file for file in os.listdir(self.path)]
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(os.path.join(self.path, image_fn))
            
        if self.transform is not None:
            image = self.transform(image)

        return image, image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def cal_mean_and_sd(loader):
    """Compute the mean and sd of the dataset

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
    train_data_path = '/home/rayting/Henry/DLCV/hw1/hw1_data/p1_data/train_50'
    valid_data_path = '/home/rayting/Henry/DLCV/hw1/hw1_data/p1_data/val_50'

    train_tfm = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                                ])

    # load the training set
    train_set = Data(train_data_path, transform=train_tfm)
    print('# images in train_set:', len(train_set)) # Should print 22500

    # load the validation set
    valid_set = Data(valid_data_path, transform=train_tfm)
    print('# images in valid_set:', len(valid_set)) # Should print 2500

    # load the testing set
    test_set = Data_inf(valid_data_path, transform=train_tfm)
    print('test_set sample:', test_set[0][0].shape, test_set[0][1]) 