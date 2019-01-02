import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from data import ImageFolder
from helpers import run_from_ipython
from nn import GAN
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/share/data/celeba')
parser.add_argument('--data', type=str, choices=['celeba', 'cifar-10', 'lsun-bedrooms'], default='celeba')
parser.add_argument('--mode', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp', 'gan-qp-l1', 'gan-qp-l2'], default='dcgan')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--n_iters', type=int, default=100000)
parser.add_argument('--d_iters', type=int, default=1)
parser.add_argument('--g_iters', type=int, default=1)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--lr', type=int, default=0.0002)
parser.add_argument('--b1', type=int, default=0.5)
parser.add_argument('--b2', type=int, default=0.999)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--n_samples', type=int, default=64)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--sample_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=1000)
parser.add_argument('--eval_interval', type=int, default=1000)
if run_from_ipython():
    # arguments when running in the Notebook
    args = parser.parse_args([
        # DCGAN
        '--mode', 'dcgan', 
        '--d_iters', '1', 
        '--g_iters', '2'
    ])
#     import matplotlib
#     %matplotlib inline
#     import matplotlib.pyplot as plt
    get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=3')
else:
    args = parser.parse_args()
args.betas = (args.b1, args.b2)

os.makedirs('{:s}/checkpoints'.format(args.mode), exist_ok=True)
os.makedirs('{:s}/samples'.format(args.mode), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
args.device = device
print('Device:', device, '/', '# of gpu:', n_gpu)

if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print('Manual seed:', args.seed)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def clamp_weights(net, value=0.01):
    for p in net.parameters():
        p.data.clamp_(-value, value)

def make_trainable(net, val):
    for p in net.parameters(): # reset requires_grad
        p.requires_grad = val

def loop(iterable):
    while True:
        for x in iter(iterable): yield x

def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)

transform = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = ImageFolder(args.data_path, transform)
dataloader = data.DataLoader(
    dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=args.n_workers
)
data = loop(dataloader)

gan = GAN(args)
gan.init(weights_init)
fixed_z = torch.randn(args.n_samples, args.z_dim).to(args.device)

writer = SummaryWriter('{:s}/summaries'.format(args.mode))

errD, errG = {}, {}
for it in range(args.n_iters):
    gan.train()
    make_trainable(gan.netD, True)
    for _ in range(args.d_iters):
        if args.mode == 'wgan':
            clamp_weights(gan.netD, 0.01)
        x_sample = next(data).to(args.device)
        z_sample = torch.randn(len(x_sample), args.z_dim).to(args.device)
        errD = gan.trainD(x_sample, z_sample)
    make_trainable(gan.netD, False)
    for _ in range(args.g_iters):
        x_sample = next(data).to(args.device)
        z_sample = torch.randn(len(x_sample), args.z_dim).to(args.device)
        errG = gan.trainG(x_sample, z_sample)
    add_scalar_dict(writer, errD, it+1, 'D')
    add_scalar_dict(writer, errG, it+1, 'G')
        
    if (it+1) % args.log_interval == 0:
        print(
            '[{:d}/{:d}] d_loss: {:.6f} g_loss: {:.6f}'.format(
            it+1, args.n_iters, errD['d_loss'], errG['g_loss']
        ))
    if (it+1) % args.sample_interval == 0:
        gan.eval()
        x_fake = gan.netG(fixed_z)
        vutils.save_image(x_fake, '{:s}/samples/{:06d}.jpg'.format(args.mode, it+1), nrow=8, normalize=True, range=(-1., 1.))
    if (it+1) % args.save_interval == 0:
        gan.save('{:s}/checkpoints/weights.{:06d}.pth'.format(args.mode, it+1))
