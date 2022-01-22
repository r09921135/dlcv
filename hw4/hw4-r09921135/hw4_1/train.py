import argparse
import os.path as osp
from qqdm import qqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train_way', type=int, default=10)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--save_path', default='./proto_1')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        progress_bar = qqdm(train_loader)
        for i, batch in enumerate(progress_bar, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_support, data_query = data[:p], data[p:]

            prototype = model(data_support)
            prototype = prototype.reshape(args.shot, args.train_way, -1).mean(dim=0)  # (train_way, 1600)
            query_feat = model(data_query)  # (train_way*query, 1600)

            query_label = torch.arange(args.train_way).repeat(args.query)
            query_label = query_label.type(torch.cuda.LongTensor)
            
            logits = euclidean_metric(query_feat, prototype)
            loss = F.cross_entropy(logits, query_label)
            acc = count_acc(logits, query_label)

            progress_bar.set_infos({
                'epoch': epoch,
                'loss': round(loss.item(), 4),
                'acc': round(acc, 4)
            })

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            prototype = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_support, data_query = data[:p], data[p:]  # data_support: (5, 3, 84, 84), data_query: (75, 3, 84, 84)

            prototype = model(data_support)  # (5, 1600)
            prototype = prototype.reshape(args.shot, args.test_way, -1).mean(dim=0)
            query_feat = model(data_query)  # (75, 1600)

            query_label = torch.arange(args.test_way).repeat(args.query)  # e.g [0,1,2,3,4,0,1,2,3,4,...]
            query_label = query_label.type(torch.cuda.LongTensor)  # (75)

            logits = euclidean_metric(query_feat, prototype)  # (75, 5)
            loss = F.cross_entropy(logits, query_label)
            acc = count_acc(logits, query_label)

            vl.add(loss.item())
            va.add(acc)
            
            prototype = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max_acc')
            print('Saving best model...')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch_last')

        # if epoch % args.save_epoch == 0:
        #     save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

