# GNU Makefile
ZIP=zip
RM=rm -f

all: ./zip/yolo.zip ./zip/depth.zip

clean:
	-$(RM) ./zip/yolo.zip ./zip/depth.zip

./zip/yolo.zip: ./src/yolo/yolo_utils.py ./src/yolo/yolo_net.py ./src/yolo/yolo_train.py ./src/yolo/yolo_eval.py
	$(ZIP) -r $@ $?

./zip/depth.zip: ./src/depth/depth_utils.py ./src/depth/depth_net.py ./src/depth/depth_train.py ./src/depth/depth_eval.py
	$(ZIP) -r $@ $?
