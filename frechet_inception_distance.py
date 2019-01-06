# https://github.com/mseitzer/pytorch-fid
# Revised by [elvisyjlin](https://github.com/elvisyjlin)
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from scipy import linalg
from tqdm import tqdm

from inception import InceptionV3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an 
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an 
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class FrechetInceptionDistance():
    def __init__(self, dims, gpu, resize=True):
        """ Constructor
        dims -- dimensionality of features returned by Inception
        gpu -- whether or not to run on GPU
        """
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.dims = dims
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        print('Using device:', self.device)

        # Load inception model
        self.inception_model = InceptionV3([block_idx], resize_input=resize, normalize_input=False).to(self.device).eval()
        print('Loaded pretrained weights of Inception v3.')

    def compute(self, imgs1, imgs2, batch_size=32, splits=1, shuffle=False):
        """Calculates the FID of two image sets"""
        act1 = self.get_activations(imgs1, batch_size, True)
        act2 = self.get_activations(imgs2, batch_size, True)
        print('# of images:', len(act1), len(act2), ' / # of img1 split:', splits)
        print('Calculating FID...')
        
        if shuffle:
            np.random.shuffle(act1)
        N1 = len(act1)
        split_scores = []
        for k in tqdm(range(splits)):
            act1_part = act1[k * (N1 // splits): (k+1) * (N1 // splits), :]
            m1, s1 = self.calculate_activation_statistics(act1_part)
            m2, s2 = self.calculate_activation_statistics(act2)
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)  # order of 1 and 2 differs slightly
            split_scores.append(fid_value)
        
        return np.mean(split_scores), np.std(split_scores)

    def calculate_activation_statistics(self, act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def get_activations(self, imgs, batch_size=64, verbose=False):
        to_save = False
        if type(imgs) is str:
            act_file = os.path.join('fid', imgs + '.act.npy')
            if os.path.exists(act_file):
                if verbose:
                    print('Loading activations from pre-calculated vectors...')
                return np.load(act_file)
            
            from custom_helpers import get_dataset
            to_save = True
            dataset_name, img_size = imgs.rsplit('.', 1)
            imgs = get_dataset(dataset_name, int(img_size))
        
        N = len(imgs)
        assert batch_size > 0
        assert N > batch_size
        
        dataloader = data.DataLoader(imgs, batch_size=batch_size)

        pred_arr = np.zeros((N, self.dims))
        for i, batch in enumerate(dataloader):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, len(dataloader)), end='', flush=True)
            batch = batch.to(self.device)
            batch_size_i = batch.size()[0]
            start = i * batch_size
            end = start + batch_size_i

            with torch.no_grad():
                pred = self.inception_model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.data.cpu().numpy().reshape(batch_size_i, -1)
        
        if to_save:
            os.makedirs('fid', exist_ok=True)
            np.save(act_file, pred_arr)
            if verbose:
                print('\rSaved activations as', act_file)

        if verbose:
            print(' done')

        return pred_arr


if __name__ == '__main__':
    FID = FrechetInceptionDistance(dims=2048, gpu=True, resize=True)
    
    print("Calculating Frechet Inception Distance for CIFAR-10 training set and validation set...")
    print(FID.compute(
#         'cifar-10.valid.32', 'cifar-10.train.32', 
        'cifar-10.train.32', 'cifar-10.valid.32', 
        batch_size=64, splits=10, shuffle=True
    ))