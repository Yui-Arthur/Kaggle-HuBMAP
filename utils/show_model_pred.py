import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import gc
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
from pathlib import Path
import math
import cv2

from gen_mask import get_valid_data
from show_mask import show_binary_mask



def show_model_pred(model , image_path , device , tfm , threshold=0.5):
    img = Image.open(image_path)
    img = tfm(img)
    rel = model(img.unsqueeze(0).to(device))[0]      

    
    masks = rel["masks"].to("cpu").detach().numpy()
    boxes = rel["boxes"].to("cpu").detach().numpy()
    score = rel["scores"].to("cpu").detach().numpy()

    # img = np.ones((1331,2000,3) , dtype=np.uint8) * 255
    show_coco_mask(image_path, masks , boxes , score , threshold)
    # cv2.waitKey(0)

def show_coco_mask(image_path , masks , boxes , score , threshold):
    img = cv2.imread(str(image_path))
    
    print(len(masks))
    for idx in range(len(masks)):
        # print(f"{idx} : {score[idx]}")
        if score[idx] < threshold:
            continue

        mask = (masks[idx][0] * 255).astype(np.uint8)
        mask = np.expand_dims(mask , axis = 2)
        mask = np.concatenate([mask, mask , mask] , axis = 2)
        
        img = cv2.bitwise_or(img , mask)

        
        
        box = boxes[idx]
        # print(box)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0,0,255), 2)

    cv2.imshow("img" , img)

if __name__ == "__main__":
    def get_model_instance_segmentation(num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 1024
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
        return model

    ROOT = Path("hubmap-hacking-the-human-vasculature")
    polygon_file = "polygons.jsonl"
    image_folder = "train"
    metadata_file = "tile_meta.csv"
    labels = pd.read_json(ROOT / "polygons.jsonl" , lines=True)
    matadata = pd.read_csv(ROOT / "tile_meta.csv")
    id_list = get_valid_data(labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model_instance_segmentation(2).to(device)
    model.load_state_dict(torch.load(f"models/mask-rcnn-model_best_1024.ckpt"))
    model.eval()
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ]) 
    


    for file in (ROOT / "train").glob("*.tif"):
        print(file.stem)
        # print(labels[labels['id'] == file.stem]['annotations'].values[0])
        if file.stem in id_list:
            show_model_pred(model , file ,device , tfm , 0.3)
            show_binary_mask(ROOT , file.stem , labels[labels['id'] == file.stem]['annotations'].values[0] , 0 , 0)
    # img = Image.open("test.jpg")

    # img = tfm(img)
    # rel = model(img.unsqueeze(0).to(device))[0]
    # print(rel["boxes"][0])
    
    # masks = rel["masks"].to("cpu").detach()#.numpy()
    # boxes = rel["boxes"].to("cpu").detach()#.numpy()

    # # img = np.ones((1331,2000,3) , dtype=np.uint8) * 255
    # img = cv2.imread("test.jpg")
    # img = cv2.resize(img , (1000,665))

    # for idx in range(len(masks)):
    #     mask = (masks[idx][0].numpy() * 255).astype(np.uint8)
    #     mask = np.expand_dims(mask , axis = 2)
    #     mask = np.concatenate([mask, mask , mask] , axis = 2)
    #     print(mask.nonzero())
    #     # print(mask.shape)
    #     # print(mask.dtype)
    #     # print(img.shape)
    #     # print(img.dtype)
    #     # img = cv2.bitwise_xor(img , mask)
    #     img = cv2.bitwise_or(img , mask)
        
    #     box = boxes[idx].numpy()
    #     # print(box)
    #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0,0,255), 1)

    # cv2.imshow("img" , img)
    # cv2.waitKey(0)
    pass