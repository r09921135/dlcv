import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from dataset import *
from model import Model


def test(args):
    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    # load the validation set
    test_set = Data_office_inf(args.test_dir, args.csv_dir, transform=test_tfm)
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
    print('Loading model...')
    model = Model().to(device)
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint)
    model.eval()

    # evaluating
    print('Start evaluating!')
    with torch.no_grad():
        names = []
        pred_total = []
        for idx, (data, name) in enumerate(test_loader):
            data = data.to(device)

            output = model(data)

            pred = output.argmax(dim=-1).cpu()
            pred_total.append(pred.numpy())
            names.append(name[0])
    
    category_dict = np.load('./hw4_2/category_dict.npy',allow_pickle='TRUE').item()
    with open((args.out_dir), 'w') as f:
        f.write('id,filename,label\n')
        for i, y in enumerate(pred_total):
            category = categoryDict_reverse(int(y), category_dict)
            f.write('{},{},{}\n'.format(i, names[i], category))
            
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./hw4_2/0.4039.pth.tar', 
                    help='save model directory', type=str)
    parser.add_argument('--csv_dir', default='/home/rayting/Henry/DLCV/hw4/hw4_data/office/val.csv', 
                    help='test csv directory', type=str)
    parser.add_argument('--test_dir', default='/home/rayting/Henry/DLCV/hw4/hw4_data/office/val', 
                    help='test images directory', type=str)
    parser.add_argument('--out_dir', default='./pred.csv', 
                    help='output images directory', type=str)       
    args = parser.parse_args()

    test(args)