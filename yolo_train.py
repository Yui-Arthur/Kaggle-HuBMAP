from ultralytics import YOLO

import os
os.chdir("yolo_coco/")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load a model

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)


# Train the model
model.train(data='kidney.yaml', epochs=10, imgsz=512, workers=0)