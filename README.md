# Evaluation of Generative Adversarial Networks

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
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode dcgan --d_iters 1 --g_iters 2
```

WGAN

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode wgan --d_iters 5 --g_iters 1
```

WGAN-GP

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode wgan-gp --d_iters 5 --g_iters 1
```

GAN-QP-L1

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode gan-qp-l1 --d_iters 2 --g_iters 1
```

GAN-QP-L2

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode gan-qp-l2 --d_iters 2 --g_iters 1
```
