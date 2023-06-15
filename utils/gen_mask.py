from pathlib import Path
import pandas as pd    
import cv2
import torch
import numpy as np 
import json



def gen_mask(annotations):
    boxes = []
    masks = []
    # area = []
    for i in range(len(annotations)):

        if annotations[i]['type'] == "blood_vessel":
            pts = np.array(annotations[i]['coordinates'])
            # print(pts[0][0])
            min_xy = np.min(pts, axis=1)[0]
            max_xy = np.max(pts, axis=1)[0]

            boxes += [np.concatenate((min_xy, max_xy), axis=0)]

            mask = np.zeros((512,512), dtype=np.uint8)
            cv2.fillPoly(mask, pts, 1)

            # area.append((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]))

            masks += [mask]
        # break
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    return boxes , masks , area

def get_valid_data(labels):
    valid_data = []
    for i in range(len(labels)):
        for j in range(len(labels.iloc[i]['annotations'])):
            if labels.iloc[i]['annotations'][j]['type'] == "blood_vessel":
                valid_data.append(labels.iloc[i]['id'])
                break
    
    return valid_data


def gen_KidneyDataset(name , labels , metadata , image_list):
        
    image_with_target = {}

    image_path = Path(image_list[0]).parent
    print(image_path)
    
    for idx , col in labels.iterrows():
        
        if f"{image_path}\\{col['id']}.tif" in image_list:
        # if f"{image_path}/{col['id']}.tif" in image_list:
            polygons = col['annotations']
            # print(polygons[0])
            boxes , masks , area = gen_mask(polygons)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes.tolist()
            target["labels"] = torch.zeros((len(boxes),), dtype=torch.int64).tolist()
            target["masks"] = masks.tolist()
            target["image_id"] = torch.tensor([idx]).tolist()
            target["area"] = area.tolist()
            target["iscrowd"] = iscrowd.tolist()
            image_with_target[col['id']] = target
        # break

    print(image_with_target)    
    with open(f"{name}.json", "w") as outfile:
        json.dump(image_with_target, outfile)


if __name__ == '__main__':
    ROOT = Path("hubmap-hacking-the-human-vasculature")
    labels = pd.read_json(ROOT / "polygons.jsonl" , lines=True)
    id_list = get_valid_data(labels)
    metadata = pd.read_csv(ROOT / "tile_meta.csv")
    image_folder = ROOT  / "train"

    train_valid_ratio = 0.9
    image_list = [str(i) for i  in (image_folder).glob('*.tif') if i.stem in id_list]
    train_list = image_list[:int(len(image_list)*train_valid_ratio)]    
    valid_list = image_list[int(len(image_list)*train_valid_ratio):]

    gen_KidneyDataset(f"{train_valid_ratio}_train" , labels , metadata , train_list)
    gen_KidneyDataset(f"{1-train_valid_ratio}_valid" , labels , metadata , valid_list)

    # print(id_list)
    # for i in range(len(labels)):
    #     # print(i)
    #     if labels.iloc[i]['id'] in id_list:
    #         boxes , masks , area = gen_mask(labels.iloc[i]['annotations'])
    #         print(torch.count_nonzero(masks))
        
    #     break
