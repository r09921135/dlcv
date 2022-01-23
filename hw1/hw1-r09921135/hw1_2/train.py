import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from skimage.transform import resize
import torch.optim as optim
import argparse
from dataset import *
from model import *
from mean_iou_evaluate import mean_iou_score

def train(args):
    # calculate the mean and std of the dataset
    # t_set = Data(args.train_dir, 'train', transform=transforms.ToTensor())
    # v_set = Data(args.val_dir, 'valid', transform=transforms.ToTensor())
    # concat_set = ConcatDataset([t_set, v_set])
    # loader = DataLoader(concat_set, batch_size=1, shuffle=False)
    # mean, std = cal_mean_and_std(loader)
    # print('Data mean:', mean, 'Data std:', std)
    mean = [0.4092, 0.3792, 0.2816]
    std = [0.1375, 0.1025, 0.0919]

    train_tfm = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    # load the training set
    train_set = Data(args.train_dir, 'train', transform=train_tfm)
    # load the validation set
    valid_set = Data(args.val_dir, 'valid', transform=train_tfm)
    print('# images in train_set:', len(train_set)) # Should print 2000
    print('# images in valid_set:', len(valid_set)) # Should print 257

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    seed = 25
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cuda' if use_cuda else 'cpu'
    print('Device used:', device)

    # load model
    model = FCN8(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print('Start training!')
    best_mean_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = []
        for idx, (data, label_batch, _) in enumerate(train_loader):
            data, label_batch = data.to(device), label_batch.to(device)
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, label_batch.long())
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        print('Train | Epoch: {}, Loss: {:.5f}'.format(epoch, train_loss))

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                labels = torch.FloatTensor()
                pred = torch.FloatTensor().cuda()
                for idx, (data, label_batch, _) in enumerate(valid_loader):
                    data = data.to(device)
                    output = model(data)

                    labels = torch.cat((labels, label_batch), 0)
                    pred = torch.cat((pred, output), 0)

                labels = labels.numpy()
                pred = pred.cpu().numpy()
                pred = np.argmax(pred, 1)
                # upsample predicted masks
                pred = np.array([resize(p, output_shape=(512, 512), order=0, preserve_range=True, clip=True) for p in pred]) 

                # calulate mIOU   
                mean_iou = mean_iou_score(pred, labels)
                
                # save model
                if mean_iou > best_mean_iou and  mean_iou > 0.68:
                    best_mean_iou = mean_iou
                    checkpoint = os.path.join(args.save_dir, '{:.4f}.pth.tar'.format(mean_iou))
                    torch.save(model.state_dict(), checkpoint)
                    print('Saving model...')

                print('Valid | Epoch: {}, mean_iou: {:.5f}'.format(epoch, mean_iou))

                if epoch in [5, 10]:
                    # save predicted results
                    print("Saving images...")
                    idx_list = [10, 97, 107]
                    mask_rgb = np.empty((512, 512, 3))
                    for i in idx_list:
                        mask_rgb[pred[i] == 0] = [0,255,255]
                        mask_rgb[pred[i] == 1] = [255,255,0]
                        mask_rgb[pred[i] == 2] = [255,0,255]
                        mask_rgb[pred[i] == 3] = [0,255,0]
                        mask_rgb[pred[i] == 4] = [0,0,255]
                        mask_rgb[pred[i] == 5] = [255,255,255]
                        mask_rgb[pred[i] == 6] = [0,0,0]
                        mask_rgb = mask_rgb.astype(np.uint8)
                        img = PIL.Image.fromarray(mask_rgb)
                        img.save(os.path.join(args.out_dir, "{:04d}_mask_ep_{}.png".format(i, epoch))) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=35, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--train_dir', default='/home/rayting/Henry/DLCV/hw1/hw1_data/p2_data/train/', 
                    help='training images directory', type=str)
    parser.add_argument('--val_dir', default='/home/rayting/Henry/DLCV/hw1/hw1_data/p2_data/validation/', 
                    help='validation images directory', type=str)
    parser.add_argument('--save_dir', default='.', 
                    help='save model directory', type=str)
    parser.add_argument('--out_dir', default='./exp', 
                    help='output images directory', type=str)
    args = parser.parse_args()

    train(args)

    

