if [ ! -d vid ]; then
  mkdir vid
fi

for D in `find out -type d -maxdepth 1`
do
  if [ -d "$D/samples" ] && [ ! -f "vid/$D.mp4" ]; then
    python3 make_video.py -i "$D/samples" -o "vid/$D.mp4" -wh
  fi
done