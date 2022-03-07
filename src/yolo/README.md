# YOLO

Training:

  $ python yolo_train.py --epochs 100 --save-model ./yolo_model.pt path/to/PASCALVOC2007.zip

Evaluation with VOC test set:

  $ python yolo_eval.py ./yolo_model.pt path/to/PASCALVOC2007.zip

Detection with images:

  $ python yolo_eval.py ./yolo_model.pt image1.jpg image2.jpg ...

Detection with camera:

  $ python yolo_eval.py ./yolo_model.pt --camera
