import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from skimage.transform import resize
import torch.optim as optim
import argparse
from dataset import *
from model import FCN8
from mean_iou_evaluate import mean_iou_score

def test(args):
    mean = [0.4092, 0.3792, 0.2816]
    std = [0.1375, 0.1025, 0.0919]
    test_tfm = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    # load the validation set
    if args.with_iou:
        test_set = Data(args.test_dir, 'valid', transform=test_tfm)
    else:
        test_set = Data_inf(args.test_dir, transform=test_tfm)
    print('# images in test_set:', len(test_set)) 

    # create dataloader
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    seed = 25
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cuda' if use_cuda else 'cpu'
    print('Device used:', device)

    # load model
    print('Loading model!')
    model = FCN8().to(device)
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint)

    print('Start inferencing!')
    model.eval()
    if args.with_iou:
        with torch.no_grad():
            labels = torch.FloatTensor()
            pred = torch.FloatTensor().cuda()
            names = []
            for idx, (data, label_batch, name) in enumerate(test_loader):
                data = data.to(device)
                pred_batch = model(data)

                labels = torch.cat((labels, label_batch), 0)
                pred = torch.cat((pred, pred_batch), 0)
                names.append(name[0])

            labels = labels.numpy()
            pred = pred.cpu().numpy()
            pred = np.argmax(pred, 1)
            # upsample predicted masks
            pred = np.array([resize(p,output_shape=(512, 512), order=0, preserve_range=True, clip=True) for p in pred]) 

            # calulate mean_iou   
            mean_iou = mean_iou_score(pred, labels)
            print('mean_iou: {:.5f}'.format(mean_iou))

    else:
        with torch.no_grad():
            pred = torch.FloatTensor().cuda()
            names = []
            for idx, (data, name) in enumerate(test_loader):
                data = data.to(device)
                pred_batch = model(data)

                pred = torch.cat((pred, pred_batch), 0)
                names.append(name[0])

            pred = pred.cpu().numpy()
            pred = np.argmax(pred, 1)
            # upsample predicted masks
            pred = np.array([resize(p,output_shape=(512, 512), order=0, preserve_range=True, clip=True) for p in pred]) 

        # save predicted results
        n_masks = len(pred)
        masks_rgb = np.empty((n_masks, 512, 512, 3))
        for i, p in enumerate(pred):
            masks_rgb[i, p == 0] = [0,255,255]
            masks_rgb[i, p == 1] = [255,255,0]
            masks_rgb[i, p == 2] = [255,0,255]
            masks_rgb[i, p == 3] = [0,255,0]
            masks_rgb[i, p == 4] = [0,0,255]
            masks_rgb[i, p == 5] = [255,255,255]
            masks_rgb[i, p == 6] = [0,0,0]
        masks_rgb = masks_rgb.astype(np.uint8)
        print("Saving image...")
        for i, mask_rgb in enumerate(masks_rgb):
            img = PIL.Image.fromarray(mask_rgb)
            img.save(os.path.join(args.out_dir, '{}.png'.format(names[i])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./0.7029.pth.tar', 
                    help='save model directory', type=str)
    parser.add_argument('--test_dir', default='/home/rayting/Henry/DLCV/hw1/hw1_data/p2_data/validation', 
                    help='test images directory', type=str)
    parser.add_argument('--out_dir', default='./output', 
                    help='output images directory', type=str)
    parser.add_argument('--with_iou', action='store_true')
    args = parser.parse_args()

    test(args)