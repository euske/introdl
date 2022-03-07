#!/bin/sh
exec >>train_depth.log
exec 2>&1
exec </dev/null

echo
date +'*** START %Y-%m-%d %H:%M:%S ***'
renice -n +20 -p $$
python depth_train.py --epochs 200 --model-coarse ./depth_model_coarse.pt ~/data/NYUDEPTH/nyu_depth_v2_labeled.mat
python depth_train.py --train-fine --epochs 200 --model-coarse ./depth_model_coarse.pt --model-fine ./depth_model_fine.pt ~/data/NYUDEPTH/nyu_depth_v2_labeled.mat
