from ultralytics import YOLO

import os
os.chdir("yolo_coco/")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load a model

model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)


# Train the model
dataset_name = "train_1_valid_1"
model.train(data=f'{dataset_name}/kidney.yaml', epochs=30, imgsz=512, workers=0)