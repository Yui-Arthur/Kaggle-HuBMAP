from pathlib import Path
import pandas as pd    
import cv2
import torch
import numpy as np 




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
    print(area)
    print(boxes)
    print(masks)

def get_valid_data(labels):
    valid_data = []
    for i in range(len(labels)):
        for j in range(len(labels.iloc[i]['annotations'])):
            if labels.iloc[i]['annotations'][j]['type'] == "blood_vessel":
                valid_data.append(labels.iloc[i]['id'])
                break
    
    return valid_data
    print(valid_data)
if __name__ == '__main__':
    ROOT = Path("hubmap-hacking-the-human-vasculature")
    labels = pd.read_json(ROOT / "polygons.jsonl" , lines=True)
    id_list = get_valid_data(labels)
    print("359bb86fa14" in id_list)
    print(labels[labels['id'] == "359bb86fa14"])
    # print(id_list)
    # for i in range(len(labels)):
    #     # print(i)
    #     if labels.iloc[i]['id'] in id_list:
    #         gen_mask(labels.iloc[i]['annotations'])
    #     # break
