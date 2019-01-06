command="env CUDA_VISIBLE_DEVICES=$1 python3 train.py --gpu"

if [ "$2" = "dcgan" ]; then
  command="$command --mode dcgan --d_iters 1 --g_iters 2"
elif [ "$2" = "wgan" ]; then
  command="$command --mode wgan --d_iters 5 --g_iters 1"
elif [ "$2" = "lsgan" ]; then
  command="$command --mode lsgan --d_iters 1 --g_iters 1"
elif [ "$2" = "wgan-gp" ]; then
  command="$command --mode wgan-gp --d_iters 5 --g_iters 1"
elif [ "$2" = "lsgan-gp" ]; then
  command="$command --mode lsgan-gp --d_iters 1 --g_iters 1"
elif [ "$2" = "dragan" ]; then
  command="$command --mode dragan --d_iters 1 --g_iters 1"
elif [ "$2" = "gan-qp-l1" ]; then
  command="$command --mode gan-qp-l1 --d_iters 2 --g_iters 1"
elif [ "$2" = "gan-qp-l2" ]; then
  command="$command --mode gan-qp-l2 --d_iters 2 --g_iters 1"
elif [ "$2" = "rsgan" ]; then
  command="$command --mode rsgan --d_iters 1 --g_iters 1"
else
  echo "Not supported mode: $2"
  exit 1
fi

if [ "$3" = "celeba" ]; then
  command="$command --data celeba --data_path /share/data/celeba --img_size 64"
elif [ "$3" = "lsun-bed" ]; then
  command="$command --data lsun-bed --data_path /share/data/lsun --img_size 64"
elif [ "$3" = "cifar-10" ]; then
  command="$command --data cifar-10 --data_path /share/data/cifar-10 --img_size 32"
elif [ "$3" = "imagenet-32" ]; then
  command="$command --data imagenet --data_path /share/data/imagenet/train_32x32 --img_size 32"
elif [ "$3" = "imagenet-64" ]; then
  command="$command --data imagenet --data_path /share/data/imagenet/train_64x64 --img_size 64"
else
  echo "Not supported data: $3"
  exit 1
fi

if [ ! -z "$4" ]; then
  command="$command --$4"
fi

echo $command
$command