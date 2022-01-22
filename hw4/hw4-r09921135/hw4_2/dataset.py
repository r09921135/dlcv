from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image


class Data_mini(Dataset):
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
        image = Image.open(os.path.join(self.path, image_fn)).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def categoryDict(csv_path):
    category_dict = {}
    lb = -1
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    for l in lines:
        id, name, category = l.split(',')
        if category not in category_dict.keys():
            lb += 1
            category_dict[category] = lb

    return category_dict

# category_dict = categoryDict('/home/rayting/Henry/DLCV/hw4/hw4_data/office/train.csv')
# np.save('category_dict.npy', category_dict) 


def categoryDict_reverse(val, dict):
    for key, value in dict.items():
         if val == value:
             return key


class Data_office(Dataset):
    def __init__(self, dir_root, split, transform=None):
        """ Intialize the dataset """
        self.transform = transform

        csv_path = os.path.join(dir_root, split + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        for l in lines:
            id, name, category = l.split(',')
            data_path = os.path.join(dir_root, split, name)
            data.append(data_path)
            label.append(category_dict[category])

        self.data = data
        self.label = label 
        self.len = len(self.data)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.data[index]
        label = self.label[index]

        image = Image.open(image_fn).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class Data_office_inf(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        """ Intialize the dataset """
        self.transform = transform
        self.data_dir = data_dir

        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        for l in lines:
            id, name, category = l.split(',')
            data.append(name)

        self.data = data
        self.len = len(self.data)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.data[index]

        data_path = os.path.join(self.data_dir, image_fn)
        image = Image.open(data_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len



if __name__ == '__main__':
    train_tfm = transforms.Compose([
                                transforms.Resize((128, 128)),
                                transforms.ToTensor()
                                ])

    data_path = '/home/rayting/Henry/DLCV/hw4/hw4_data/mini/train'
    # load mini_imagenet
    train_mini = Data_mini(data_path, train_tfm)
    print('# images in train_mini:', len(train_mini)) # Should print 22500

    data_root = '/home/rayting/Henry/DLCV/hw4/hw4_data/office'
    # load office_home
    train_office = Data_office(data_root, 'train', train_tfm)
    print('# images in train_office:', len(train_office))

    val_office = Data_office(data_root, 'val', train_tfm)
    print('# images in val_office:', len(val_office))
    