import torch
import torchvision
import os
import random
import argparse

from model import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--out_path', default='./output')
parser.add_argument('--log_path', default='./logs')
parser.add_argument('--model_path', default='./models/G_29.7.pth')
args = parser.parse_args()

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
# seed = random.randint(1, 10000)
seed = 299
print("Random Seed: ", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cuda' if use_cuda else 'cpu'
print('Device used:', device)

# Training hyperparameters
z_dim = 100

# Model
G = Generator()
G.load_state_dict(torch.load(
    args.model_path, map_location=torch.device(device)))
G.eval()
if device == 'cuda':
    G.cuda()

# Generate 1000 images and make a grid to save them.
n_output = 1000
z_sample = torch.randn(n_output, z_dim, 1, 1)
if device == 'cuda':
    z_sample = z_sample.cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0

# Show 32 of the images.
# grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=8)
# save_path = os.path.join(args.log_path, 'examples.jpg')
# torchvision.utils.save_image(grid_img, save_path)

# Save the generated images.
for i in range(1000):
    save_path = os.path.join(args.out_path, f'{i+1:04d}.png')
    torchvision.utils.save_image(imgs_sample[i], save_path)

print('Done!')
