# Evaluation of Generative Adversarial Networks

## Requirements

* Python 3
* PyTorch
* TensorboardX

```bash
pip3 install -r requirements.txt
```

## To Train a

DCGAN

```python
CUDA_VISIBLE_DEVICES=0 python3 train.py --mode dcgan --d_iters 1 --g_iters 2
```
