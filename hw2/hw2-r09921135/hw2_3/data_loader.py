import torch.utils.data as data
from PIL import Image
import os
import pandas as pd
import torchvision.datasets as datasets


class Data(data.Dataset):
    def __init__(self, path, dtype, transform=None):
        """ Intialize the dataset """
        self.samples = []
        data_path = os.path.join(path, dtype)
        filenames = [filename for filename in os.listdir(data_path)]
        labels = pd.read_csv(os.path.join(path, dtype + '.csv'))
        
        self.len = len(filenames)

        # read data
        for i in range(self.len):
            image_name = filenames[i]
            image = Image.open(os.path.join(data_path, image_name))
            if transform is not None:
                image = transform(image)
            label = int(labels[labels.image_name == image_name].label)
            self.samples.append((image, label))
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image = self.samples[index][0]
        label = self.samples[index][1]

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class Data_inf(data.Dataset):
    def __init__(self, path, transform=None):
        """ Intialize the dataset """
        self.samples = []
        filenames = [filename for filename in os.listdir(path)]
        filenames = sorted(filenames)
        
        self.len = len(filenames)

        # read data
        for i in range(self.len):
            image_name = filenames[i]
            image = Image.open(os.path.join(path, image_name))
            if transform is not None:
                image = transform(image)
            self.samples.append((image, image_name))
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image = self.samples[index][0]
        image_name = self.samples[index][1]

        return image, image_name

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

