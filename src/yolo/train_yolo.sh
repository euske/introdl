#!/bin/sh
exec >>train_yolo.log
exec 2>&1
exec </dev/null

echo
date +'*** START %Y-%m-%d %H:%M:%S ***'
renice -n +20 -p $$
exec python yolo_train.py --epochs 100 --save-model ./yolo_model.pt ~/data/PASCAL/VOC2007.zip
