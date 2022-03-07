# Depth

Training:

  $ python depth_train.py --epochs 100 --model-coarse ./depth_model_coarse.pt path/to/nyu_depth_v2_labeled.mat
  $ python depth_train.py --epochs 100 --train-fine --model-coarse ./depth_model_coarse.pt --model-fine ./depth_model_fine.pt path/to/nyu_depth_v2_labeled.mat

Detection with images:

  $ python depth_eval.py ./depth_model_coarse.pt ./depth_model_fine.pt image1.jpg image2.jpg ...

Detection with camera:

  $ python depth_eval.py ./depth_model_coarse.pt ./depth_model_fine.pt --camera
