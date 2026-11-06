#!/usr/bin/env bash
# Render frames START..END in chunks; on a crash (GPU reset), resume after
# the last frame saved.   ./render_chunks.sh 535 2520 [pct] [samples] [outdir]
set -u
start=$1 end=$2 pct=${3:-25} samples=${4:-16} out=${5:-frames} tries=0
mkdir -p "renders/$out"
while [ "$start" -le "$end" ]; do
  stop=$((start + 299 < end ? start + 299 : end))
  if ~/.local/bin/blender -b --factory-startup -P build_scene.py -- --pct "$pct" \
      --samples "$samples" --anim --start "$start" --end "$stop" --out "$out" > renders/chunk.log 2>&1; then
    start=$((stop + 1)) tries=0
  else
    last=$(grep -o "f_[0-9]*\.png" renders/chunk.log | tail -1 | tr -dc 0-9 | sed 's/^0*//')
    tries=$((tries + 1))
    echo "crash in $start-$stop after frame ${last:-none} (try $tries)" >&2
    [ "$tries" -ge 4 ] && exit 1
    [ -n "$last" ] && start=$((last + 1))
  fi
done
