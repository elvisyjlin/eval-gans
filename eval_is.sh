COMMAND="env CUDA_VISIBLE_DEVICES=3 python3 eval.py --eval_samples 50000 --iter -1 --gpu"

for D in `find out -type d -maxdepth 1`
do
  if [ ! -d "$D/checkpoints" ]; then
    continue
  fi
  mode="$(echo ${D:4} | cut -d'.' -f1)"
  data="$(echo ${D:4} | cut -d'.' -f2)"
  img_size="$(echo ${D:4} | cut -d'.' -f3)"
  ttur="$(echo ${D:4} | cut -d'.' -f4)"
  
  command="$COMMAND --mode $mode --data $data --img_size $img_size"
  if [ ! -z "$ttur" ]; then
    command="$command --$ttur"
  fi
  echo $command
  $command
done