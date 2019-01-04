import argparse
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from nn import GAN
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['celeba', 'cifar-10', 'lsun-bed'], default='celeba')
parser.add_argument('--mode', type=str, choices=['dcgan', 'wgan', 'lsgan', 'wgan-gp', 'lsgan-gp', 'dragan', 'gan-qp-l1', 'gan-qp-l2', 'rsgan'], default='dcgan')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--img_size', type=int, choices=[32, 64], default=64)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--iter', type=int, default=100000)
parser.add_argument('--test_samples', type=int, default=1000)
parser.add_argument('--ttur', action='store_true')
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
print(args)

path = 'out/{:s}.{:s}'.format(args.mode, args.data)
if args.ttur:
    path += '.ttur'
load_path = path + '/checkpoints'
output_path = path + '/test_samples'

if os.path.exists(output_path):
    shutil.rmtree(output_path, ignore_errors=True)
os.makedirs(output_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
n_gpu = torch.cuda.device_count()
args.device = device
print('Device:', device, '/', '# of gpu:', n_gpu)

if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print('Manual seed:', args.seed)

print(args)

model_path = os.path.join(load_path, 'weights.{:06d}.pth'.format(args.iter))

gan = GAN(args)
gan.load(model_path)
gan.eval()
print('Loaded model from', model_path)

for batch in tqdm(list(range(0, args.test_samples, args.batch_size))):
    n = min(args.batch_size, args.test_samples - batch)
    z = torch.randn(n, args.z_dim).to(args.device)
    x_fake = gan.netG(z)
    vutils.save_image(
        x_fake, 
        os.path.join(output_path, '{:03d}.jpg'.format(batch//args.batch_size)), 
        nrow = 8, normalize=True, range=(-1., 1.)
    )