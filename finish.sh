#!/usr/bin/env bash
# Grade the rendered frames and lay the chapter cards over them.
#   ./finish.sh renders/frames out.mp4
set -euo pipefail
frames=${1:-renders/frames}
out=${2:-qtpass-strata.mp4}
crf=${CRF:-18}
here=$(cd "$(dirname "$0")" && pwd)
read -r width height < <(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -of csv=s=x:p=0 "$frames/f_0001.png" | tr x ' ')
sigma=$((width / 90))
nframes=$(find "$frames" -name 'f_*.png' | wc -l)
fade_at=$(python3 -c "print(max($nframes / 24 - 2.5, 0))")
audio=${AUDIO:-$here/audio.wav}

cards=$(mktemp -d)
trap 'rm -rf "$cards"' EXIT
python3 "$here/cards.py" "$width" "$height" "$cards"

inputs=(-framerate 24 -i "$frames/f_%04d.png")
f="[0:v]format=gbrp,split[base][hi];"
f+="[hi]curves=all='0/0 0.62/0 1/1',gblur=sigma=$sigma[glow];"
f+="[base][glow]blend=all_mode=screen:all_opacity=0.55,"
f+="vignette=angle=PI/4.5,noise=alls=5:allf=t,"
f+="fade=t=in:st=0:d=1,fade=t=out:st=$fade_at:d=2.5[v0];"
n=0
while read -r png t0 dur out_at; do
  n=$((n + 1))
  inputs+=(-loop 1 -framerate 24 -t "$dur" -i "$png")
  f+="[$n:v]format=rgba,fade=t=in:st=0:d=0.9:alpha=1,"
  f+="fade=t=out:st=$out_at:d=0.9:alpha=1,setpts=PTS+$t0/TB[c$n];"
  f+="[v$((n - 1))][c$n]overlay=eof_action=pass:format=gbrp[v$n];"
done < "$cards/cards.txt"
f+="[v$n]format=yuv420p[v]"
maps=(-map "[v]")
if [ -f "$audio" ]; then
  inputs+=(-i "$audio")
  maps+=(-map "$((n + 1)):a" -c:a aac -b:a 192k -shortest)
fi

ffmpeg -hide_banner -loglevel error -y "${inputs[@]}" \
  -filter_complex "$f" "${maps[@]}" -c:v libx264 -preset slow -crf "$crf" \
  -movflags +faststart "$out"
echo "wrote $out"
