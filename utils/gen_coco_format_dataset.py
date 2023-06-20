from pathlib import Path
import pandas as pd    
import cv2
import torch
import numpy as np 
import json
from gen_mask import get_valid_data , get_dataset_data
import shutil
import random

def gen_label_file(filename , annotations):

    with open(filename, 'w') as f:
        for annotation in annotations:
            if annotation['type'] == "blood_vessel":
                lines = "0 "
                for x , y in annotation['coordinates'][0]:
                    lines += f"{x / 512} {y / 512} "
                f.write(lines + "\n")

def gen_yolo_format_data(SRC_ROOT , DST_ROOT , labels , train_list , valid_list):
    (DST_ROOT / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (DST_ROOT / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (DST_ROOT / 'valid' / 'images').mkdir(parents=True, exist_ok=True)
    (DST_ROOT / 'valid' / 'labels').mkdir(parents=True, exist_ok=True)



    for file_name in train_list:
        # print(file_name)
        # print( (DST_ROOT / 'train' / 'images'))
        shutil.copyfile(file_name, (DST_ROOT / 'train' / 'images' / file_name.name))
        
        annotations = labels[labels['id'] == file_name.stem]['annotations'].values[0]
        gen_label_file((DST_ROOT / 'train' / 'labels' / f"{file_name.stem}.txt") , annotations)

    for file_name in valid_list:
        # print(file_name)
        # print( (DST_ROOT / 'train' / 'images'))
        shutil.copyfile(file_name, (DST_ROOT / 'valid' / 'images' / file_name.name))
        
        annotations = labels[labels['id'] == file_name.stem]['annotations'].values[0]
        gen_label_file((DST_ROOT / 'valid' / 'labels' / f"{file_name.stem}.txt") , annotations)
        
        
            

    




if __name__ == '__main__':
    SRC_ROOT = Path("hubmap-hacking-the-human-vasculature")
    DST_ROOT = Path("yolo_coco")
    
    
    labels = pd.read_json(SRC_ROOT / "polygons.jsonl" , lines=True)
    metadata = pd.read_csv(SRC_ROOT / "tile_meta.csv")
    image_folder = SRC_ROOT  / "train"
    train_valid_ratio = 0.9
    shuffle = False
    dataset_no = [1,2]


    id_list = []
    for dataset in dataset_no:
        data = get_dataset_data(labels , metadata , dataset)
        random.shuffle(data)
        id_list += data

    if shuffle:
        random.shuffle(id_list)

    train_num = int(len(id_list)) * (train_valid_ratio)
    valid_num = int(len(id_list) * (1 - train_valid_ratio))


    image_list = [image_folder / f"{i}.tif" for i  in id_list]


    
    train_list = image_list[:int(len(image_list)*train_valid_ratio)]    
    valid_list = image_list[int(len(image_list)*train_valid_ratio):]

    
    dataset_name = "train_1_2_valid_1"


    gen_yolo_format_data(SRC_ROOT / 'train' , DST_ROOT / dataset_name, labels , train_list , valid_list)
    

    # print(id_list)
    # for i in range(len(labels)):
    #     # print(i)
    #     if labels.iloc[i]['id'] in id_list:
    #         boxes , masks , area = gen_mask(labels.iloc[i]['annotations'])
    #         print(torch.count_nonzero(masks))
        
    #     break
