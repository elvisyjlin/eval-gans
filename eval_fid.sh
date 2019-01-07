COMMAND="env CUDA_VISIBLE_DEVICES=2 python3 eval_fid.py --eval_samples 50000 --iter -1 --gpu"

for D in `find out -type d -maxdepth 1 | sort -n`
do
  if [ ! -d "$D/checkpoints" ]; then
    continue
  fi
  if [ -f "$D/eval/frechet_inception_distance.json" ]; then
    echo "FID of $D has been computed. Skip it."
    continue
  fi
  mode="$(echo ${D:4} | cut -d'.' -f1)"
  data="$(echo ${D:4} | cut -d'.' -f2)"
  img_size="$(echo ${D:4} | cut -d'.' -f3)"
  ttur="$(echo ${D:4} | cut -d'.' -f4)"
  eval_fid_data="$data.train.$img_size"
  
  command="$COMMAND --mode $mode --data $data --img_size $img_size --eval_fid_data $eval_fid_data"
  if [ ! -z "$ttur" ]; then
    command="$command --$ttur"
  fi
  echo $command
  $command
done
