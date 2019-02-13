# https://github.com/sbarratt/inception-score-pytorch
# Revised by [elvisyjlin](https://github.com/elvisyjlin)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

class InceptionScore():
    def __init__(self, gpu):
        """ Constructor
        gpu -- whether or not to run on GPU
        """
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        print('Using device:', self.device)

        # Load inception model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device).eval()
        print('Loaded pretrained weights of Inception v3.')

    def compute(self, imgs, gpu=True, batch_size=32, resize=False, splits=1):
        """ Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        batch_size -- batch size for feeding into Inception v3
        resize -- whether or not to resize images to 299x299
        splits -- number of splits
        """
        N = len(imgs)

        assert batch_size > 0
        assert N > batch_size

        # Set up dataloader
        dataloader = data.DataLoader(imgs, batch_size=batch_size)

        up = lambda x: F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True).to(self.device)
        def get_pred(x):
            if resize:
                x = up(x)
            with torch.no_grad():
                x = self.inception_model(x)
            return F.softmax(x, dim=1).data.cpu().numpy()

        # Get predictions
        preds = np.zeros((N, 1000))
        for i, batch in enumerate(tqdm(dataloader)):
            batch = batch.to(self.device)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

        # Now compute the mean kl-div
        split_scores = []
        for k in tqdm(range(splits)):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    from data import IgnoreLabelDataset, LimitedImageDataset, PureImageFolder
    
    IS = InceptionScore(gpu=True)

    # CIFAR-10
    cifar = dsets.CIFAR10(
        root='/share/data/cifar-10', train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    print('# of images:', len(cifar))
    print("Calculating Inception Score for CIFAR-10 training set...")
    print(IS.compute(IgnoreLabelDataset(cifar), batch_size=64, resize=True, splits=10))

#     # CIFAR-10
#     cifar = dsets.CIFAR10(
#         root='/share/data/cifar-10', train=False, download=True,
#         transform=transforms.Compose([
#             transforms.Resize(32),
#             transforms.CenterCrop(32),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     )
#     print('# of images:', len(cifar))
#     print("Calculating Inception Score for CIFAR-10 validation set...")
#     print(IS.compute(IgnoreLabelDataset(cifar), batch_size=64, resize=True, splits=10))
    
#     # ImageNet 32x32
#     imagenet = PureImageFolder(
#         root='/share/data/imagenet/valid_32x32', 
#         transform=transforms.Compose([
#             transforms.Resize(32),
#             transforms.CenterCrop(32),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     )
#     print('# of images:', len(imagenet))
#     print("Calculating Inception Score for ImageNet 32x32 validation set...")
#     print(IS.compute(IgnoreLabelDataset(imagenet), batch_size=64, resize=True, splits=10))
    
#     # ImageNet 64x64
#     imagenet = PureImageFolder(
#         root='/share/data/imagenet/valid_64x64', 
#         transform=transforms.Compose([
#             transforms.Resize(64),
#             transforms.CenterCrop(64),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     )
#     print('# of images:', len(imagenet))
#     print("Calculating Inception Score for ImageNet 64x64 validation set...")
#     print(IS.compute(IgnoreLabelDataset(imagenet), batch_size=64, resize=True, splits=10))
    
#     # CelebA
#     celeba = PureImageFolder(
#         root='/share/data/celeba', 
#         transform=transforms.Compose([
#             transforms.Resize(64),
#             transforms.CenterCrop(64),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     )
#     print('# of images:', 50000)
#     print("Calculating Inception Score for the first 50k images in CelebA 64x64 validation set...")
#     print(IS.compute(LimitedImageDataset(IgnoreLabelDataset(celeba), 50000), batch_size=64, resize=True, splits=10))
    
#     # LSUN bedroom
#     lsun_bed = dsets.LSUN(
#         root='/share/data/lsun', classes=['bedroom_train'], 
#         transform=transforms.Compose([
#             transforms.Resize(64),
#             transforms.CenterCrop(64),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     )
#     print('# of images:', len(lsun_bed))
#     print("Calculating Inception Score for LSUN bedroom training set...")
#     print(IS.compute(IgnoreLabelDataset(lsun_bed), batch_size=64, resize=True, splits=10))
    
#     # LSUN bedroom
#     lsun_bed = dsets.LSUN(
#         root='/share/data/lsun', classes=['bedroom_val'], 
#         transform=transforms.Compose([
#             transforms.Resize(64),
#             transforms.CenterCrop(64),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     )
#     print('# of images:', len(lsun_bed))
#     print("Calculating Inception Score for LSUN bedroom validation set...")
#     print(IS.compute(IgnoreLabelDataset(lsun_bed), batch_size=64, resize=True, splits=10))