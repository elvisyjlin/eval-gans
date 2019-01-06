import argparse
import json
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
from frechet_inception_distance import FrechetInceptionDistance
from nn import GAN
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['celeba', 'cifar-10', 'lsun-bed', 'imagenet'], default='celeba')
parser.add_argument('--mode', type=str, choices=['dcgan', 'wgan', 'lsgan', 'wgan-gp', 'lsgan-gp', 'dragan', 'gan-qp-l1', 'gan-qp-l2', 'rsgan'], default='dcgan')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--img_size', type=int, choices=[32, 64], default=64)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--iter', type=int, default=100000)
parser.add_argument('--eval_samples', type=int, default=50000)
parser.add_argument('--is_split', type=int, default=10)
parser.add_argument('--eval_fid_data', type=str, choices=['celeba.train.64', 'cifar-10.train.32', 'lsun-bed.train.64', 'imagenet.train.32', 'imagenet.train.64'])
parser.add_argument('--ttur', action='store_true')
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
print(args)

path = 'out/{:s}.{:s}.{:d}'.format(args.mode, args.data, args.img_size)
if args.ttur:
    path += '.ttur'
load_path = path + '/checkpoints'
eval_path = path + '/eval'
os.makedirs(eval_path, exist_ok=True)

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


FID = FrechetInceptionDistance(dims=2048, gpu=args.gpu, resize=True)
gan = GAN(args)

if args.iter > 0:
    model_paths = [os.path.join(load_path, 'checkpoint.{:06d}.pth'.format(args.iter))]
else:
    model_paths = []
    iter = 1000
    while os.path.exists(os.path.join(load_path, 'checkpoint.{:06d}.pth'.format(iter))):
        model_paths.append(os.path.join(load_path, 'checkpoint.{:06d}.pth'.format(iter)))
        iter += 1000

eval_result = {}
for model_path in model_paths:
    gan.load(model_path)
    gan.eval()
    print('Loaded model from', model_path)

    x_fake = []
    for batch in tqdm(list(range(0, args.eval_samples, args.batch_size))):
        n = min(args.batch_size, args.eval_samples - batch)
        z = torch.randn(n, args.z_dim).to(args.device)
        with torch.no_grad():
            x_fake.append(gan.netG(z).data.cpu().numpy())
    x_fake = np.concatenate(x_fake)

    print("Calculating Frechet Inception Distance...")
    fid_mean, fid_std = FID.compute(x_fake, args.eval_fid_data, batch_size=64, splits=10)
    print(fid_mean, fid_std)
    eval_result[model_path] = (fid_mean, fid_std)

json.dump(eval_result, open(os.path.join(eval_path, 'frechet_inception_distance.json'), 'w', encoding='utf-8'))