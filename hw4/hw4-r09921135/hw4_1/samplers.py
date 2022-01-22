import torch
import numpy as np

from mini_imagenet import MiniImageNet


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_sample_per_cls):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_sample_per_cls = n_sample_per_cls

        label = np.array(label)
        self.class_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.class_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):  # n_batch = episode
            batch = []
            classes = torch.randperm(len(self.class_ind))[:self.n_cls]  # randomly sample categories
            for c in classes:
                ind = self.class_ind[c]
                data = torch.randperm(len(ind))[:self.n_sample_per_cls]  # randomly sample instances within the same category
                batch.append(ind[data])
            batch = torch.stack(batch)  # (5, 16)
            batch = batch.t().reshape(-1)   # batch.t().shape = (16, 5)
            yield batch  # (80), return the indices of data in the batch, where the categorical order is (0,1,2,3,4,0,1,2,3,4...)


if __name__ == '__main__':
    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100, 5, 16)

    n = 0
    for i in iter(train_sampler):
        n+=1
    print('# of episodes:', n)
