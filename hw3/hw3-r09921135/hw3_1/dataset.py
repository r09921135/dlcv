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
        num_class = 37

        # read filenames
        for i in range(num_class):
            filenames = [filename for filename in os.listdir(self.path) if filename.startswith(str(i) + '_')]
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(os.path.join(self.path, image_fn))
        
        if len(image.getbands()) == 4:
            image = transforms.Resize((384, 384))(image)
            image = transforms.ToTensor()(image)
            image = image[:3]
            image = transforms.Normalize(0.5, 0.5)(image)
        else:
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

        # read filenames
        self.filenames = [file for file in os.listdir(self.path)]
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(os.path.join(self.path, image_fn))
            
        if len(image.getbands()) == 4:
            image = transforms.Resize((384, 384))(image)
            image = transforms.ToTensor()(image)
            image = image[:3]
            image = transforms.Normalize(0.5, 0.5)(image)
        else:
            if self.transform is not None:
                image = self.transform(image)

        return image, image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len



if __name__ == '__main__':
    train_data_path = '/home/rayting/Henry/DLCV/hw3/hw3_data/p1_data/train'
    valid_data_path = '/home/rayting/Henry/DLCV/hw3/hw3_data/p1_data/val'

    train_tfm = transforms.Compose([
                                transforms.Resize((384, 384)),
                                transforms.ToTensor()
                                ])

    # load the training set
    train_set = Data(train_data_path, transform=train_tfm)
    print('# images in train_set:', len(train_set)) # Should print 22500

    # load the validation set
    valid_set = Data(valid_data_path, transform=train_tfm)
    print('# images in valid_set:', len(valid_set)) # Should print 2500

    # load the testing set
    # test_set = Data_inf(valid_data_path, transform=train_tfm)
    # print('test_set sample:', test_set[0][0].shape, test_set[0][1]) 