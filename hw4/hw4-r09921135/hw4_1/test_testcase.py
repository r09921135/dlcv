import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

from convnet import Convnet
from utils import euclidean_metric

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]  # such label should be converted later
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

# This sampler can make sure that in the same batch, all the samples are from the same episode.
class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()  # episode_df.values.shape = (600, 80)

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, model, data_loader):

    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            '''
            data: (80, 3, 84, 84)
            target: (80)
            '''
            data = data.cuda()

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:]
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            # label_encoverter: e.g {'n02981792': 0, 'n03535780': 1, ...}
            label_converter = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_converter[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # extract the feature of support and query data
            proto = model(support_input)
            query_feat = model(query_input)

            # calculate the prototype for each class according to its support data
            proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0)

            # classify the query data depending on the its distense with each prototype
            logits = euclidean_metric(query_feat, proto)
            pred = torch.argmax(logits, dim=1)
            pred = pred.cpu().numpy().tolist()
            prediction_results.append(pred)

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', default='./hw4_1/model/0.4411.pth', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', default='../hw4_data/mini/val.csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', default='../hw4_data/mini/val', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', default='../hw4_data/mini/val_testcase.csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', default='./pred.csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # load your model
    model = Convnet()
    model.load_state_dict(torch.load(args.load))
    model.cuda().eval()

    prediction_results = predict(args, model, test_loader)

    # output your prediction to csv
    with open((args.output_csv), 'w') as f:
        writer = csv.writer(f)

        header = ['episode_id']
        for i in range(args.N_query * args.N_way):
            header.append(f'query{i}')
        writer.writerow(header)

        for i, pred in enumerate(prediction_results):
            row = [i]
            row.extend(pred)
            writer.writerow(row)

    print('Done!')