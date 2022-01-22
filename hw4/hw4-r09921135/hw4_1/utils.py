import os
import shutil
import time
import pprint

import torch


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)  # (train_way*query, train_way, 1600)
    b = b.unsqueeze(0).expand(n, m, -1)
    distance = ((a - b)**2).sum(dim=2)  # (train_way*query, train_way), [i, j] represents the distance between ith query and jth prototype
    logits = -distance  # try to maximize the -distance bwtween query and its corresponding prototype
    return logits


def cosSimilarity_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)  # (train_way*query, train_way, 1600)
    b = b.unsqueeze(0).expand(n, m, -1)
    cos = torch.nn.CosineSimilarity(dim=-1)
    distance = cos(a, b)
    logits = -distance  
    return logits


def parametric_metric(a, b, func):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)  # (train_way*query, train_way, 1600)
    b = b.unsqueeze(0).expand(n, m, -1)
    input = torch.abs(a - b)
    distance = func(input).squeeze(2)
    logits = -distance  
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

