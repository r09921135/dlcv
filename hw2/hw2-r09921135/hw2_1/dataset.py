from torch.utils.data import Dataset
import os
from PIL import Image


class Data(Dataset):
    def __init__(self, path, transform=None):
        """ Intialize the dataset """
        self.path = path
        self.transform = transform
        self.filenames = []

        self.filenames = [filename for filename in os.listdir(self.path)]
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_name = self.filenames[index]
        image = Image.open(os.path.join(self.path, image_name))
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
