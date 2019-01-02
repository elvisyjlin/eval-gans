# An Evaluation of Generative Adversarial Networks

An evaluation of recent variants of generative adversarial networks by PyTorch

## Requirements

* Python 3
* PyTorch
* TensorboardX

```bash
pip3 install -r requirements.txt
```


## To Start the Tensorboard

```bash
tensorboard --logdir=.
```


## To Train a

DCGAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode dcgan --data celeba --d_iters 1 --g_iters 2 --gpu --ttur
```

WGAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode wgan --data celeba --d_iters 5 --g_iters 1 --gpu --ttur
```

LSGAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode lsgan --data celeba --d_iters 1 --g_iters 1 --gpu --ttur
```

WGAN-GP

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode wgan-gp --data celeba --d_iters 5 --g_iters 1 --gpu --ttur
```

LSGAN-GP

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode lsgan-gp --data celeba --d_iters 1 --g_iters 1 --gpu --ttur
```

GAN-QP-L1

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode gan-qp-l1 --data celeba --d_iters 2 --g_iters 1 --gpu --ttur
```

GAN-QP-L2

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode gan-qp-l2 --data celeba --d_iters 2 --g_iters 1 --gpu --ttur
```
