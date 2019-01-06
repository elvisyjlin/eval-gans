
# coding: utf-8

# In[ ]:


# https://github.com/cedrickchee/wasserstein-gan
# https://blog.csdn.net/qq_25737169/article/details/78857788
# https://github.com/xudonmao/improved_LSGAN/blob/master/difficult_architectures/others/gan_64x64.py
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py


# In[ ]:


import argparse
import os
import shutil
from helpers import run_from_ipython

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/share/data/celeba')
parser.add_argument('--data', type=str, choices=['celeba', 'cifar-10', 'lsun-bed', 'imagenet'], default='celeba')
parser.add_argument('--mode', type=str, choices=['dcgan', 'wgan', 'lsgan', 'wgan-gp', 'lsgan-gp', 'dragan', 'gan-qp-l1', 'gan-qp-l2', 'rsgan'], default='dcgan')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--n_iters', type=int, default=100000)
parser.add_argument('--d_iters', type=int, default=1)
parser.add_argument('--g_iters', type=int, default=1)
parser.add_argument('--img_size', type=int, choices=[32, 64], default=64)
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
parser.add_argument('--ttur', action='store_true')
parser.add_argument('--gpu', action='store_true')
if run_from_ipython():
    # arguments when running in the Notebook
    args = parser.parse_args([
#         # CelebA
#         '--data_path', '/share/data/celeba', 
#         '--data', 'celeba', 
#         '--img_size', '64', 
#         # LSUN bedroom
#         '--data_path', '/share/data/lsun', 
#         '--data', 'lsun-bedroom', 
#         '--img_size', '64', 
#         # CIFAR-10
#         '--data_path', '/share/data/cifar-10', 
#         '--data', 'cifar-10', 
#         '--img_size', '32', 
        # ImageNet-32
        '--data_path', '/share/data/imagenet/train_32x32', 
        '--data', 'imagenet', 
        '--img_size', '32', 
#         # ImageNet-64
#         '--data_path', '/share/data/imagenet/train_64x64', 
#         '--data', 'imagenet', 
#         '--img_size', '64', 
        
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
print(args)

output_path = 'out/{:s}.{:s}.{:d}'.format(args.mode, args.data, args.img_size)
if args.ttur:
    output_path += '.ttur'

if os.path.exists(output_path):
    shutil.rmtree(output_path, ignore_errors=True)
os.makedirs('out', exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs('{:s}/checkpoints'.format(output_path), exist_ok=True)
os.makedirs('{:s}/samples'.format(output_path), exist_ok=True)


# In[ ]:


if args.ttur:
    args.g_lr = args.lr * args.g_iters / args.g_iters
    args.d_lr = args.lr * args.d_iters / args.g_iters
    args.g_iters = 1
    args.d_iters = 1
else:
    args.g_lr = args.lr
    args.d_lr = args.lr
args.g_betas = (args.b1, args.b2)
args.d_betas = (args.b1, args.b2)


# In[ ]:


import torch

device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
n_gpu = torch.cuda.device_count()
args.device = device
print('Device:', device, '/', '# of gpu:', n_gpu)


# In[ ]:


import numpy as np
import random
import torch

if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print('Manual seed:', args.seed)


# In[ ]:


print(args)


# In[ ]:


import torch.nn as nn

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def make_trainable(net, val):
    for p in net.parameters(): # reset requires_grad
        p.requires_grad = val

def loop(iterable):
    while True:
        for x, _ in iter(iterable): yield x

def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)


# In[ ]:


import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from data import PureImageFolder

transform = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
if args.data == 'celeba':
    dataset = PureImageFolder(args.data_path, transform=transform)
if args.data == 'cifar-10':
    dataset = dsets.CIFAR10(args.data_path, train=True, transform=transform, download=True)
if args.data == 'lsun-bed':
    dataset = dsets.LSUN(args.data_path, classes=['bedroom_train'], transform=transform)
if args.data == 'imagenet':
    dataset = PureImageFolder(args.data_path, transform=transform)
print('# of images in training set:', len(dataset))
dataloader = data.DataLoader(
    dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=args.n_workers
)
print('# of batches per epoch:', len(dataloader))
data = loop(dataloader)


# In[ ]:


# data = datasets.LSUN(root='data', classes=['bedroom_train'], transform=transform)


# In[ ]:


# import torchvision.utils as vutils

# for x, _ in dataloader:
#     print('Size:', x.size(), '/ Type:', x.dtype, '/ Min:', x.min(), '/ Max:', x.max())
#     vutils.save_image(x, 'test.jpg', nrow=8, normalize=True, range=(-1., 1.))
#     print('Saved the first batch of images in the dataset as test.jpg.')
#     break


# In[ ]:


from nn import GAN
from tensorboardX import SummaryWriter

gan = GAN(args)
gan.init(weights_init)
fixed_z = torch.randn(args.n_samples, args.z_dim).to(args.device)

writer = SummaryWriter('{:s}/summaries'.format(output_path))


# In[ ]:


import torchvision.utils as vutils

errD, errG = {}, {}
for it in range(args.n_iters):
    gan.train()
    # make_trainable(gan.netD, True)
    for _ in range(args.d_iters):
        x_sample = next(data).to(args.device)
        z_sample = torch.randn(len(x_sample), args.z_dim).to(args.device)
        errD = gan.trainD(x_sample, z_sample)
    # make_trainable(gan.netD, False)
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
        vutils.save_image(x_fake, '{:s}/samples/{:06d}.jpg'.format(output_path, it+1), nrow=8, normalize=True, range=(-1., 1.))
    if (it+1) % args.save_interval == 0:
        gan.save('{:s}/checkpoints/checkpoint.{:06d}.pth'.format(output_path, it+1), options=['netG'])
        gan.save('{:s}/checkpoints/checkpoint.latest.pth'.format(output_path, it+1), data={
            'iter': it+1, 'seed': args.seed, 
            'g_lr': args.g_lr, 'd_lr': args.d_lr, 
            'g_betas': args.g_betas, 'd_betas': args.d_betas
        })

