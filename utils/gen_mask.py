from pathlib import Path
import pandas as pd    
import cv2
import numpy as np 




def gen_mask(annotations):
    boxes = []
    masks = []
    for i in range(len(annotations)):

        if annotations[i]['type'] == "blood_vessel":
            pts = np.array(annotations[i]['coordinates'])
            min_xy = np.min(pts, axis=1)[0]
            max_xy = np.max(pts, axis=1)[0]
            boxes += [min_xy.tolist() + max_xy.tolist()]
            # + max_xy.tolist()


            # print(boxes)
            # print(max_xy - min_xy)
            mask = np.zeros((512,512), dtype=np.uint8)
            cv2.fillPoly(mask, pts, 1)

            masks += [mask]

    print(masks)



if __name__ == '__main__':
    ROOT = Path("hubmap-hacking-the-human-vasculature")
    labels = pd.read_json(ROOT / "polygons.jsonl" , lines=True)
    for i in range(len(labels)):
        gen_mask(labels.iloc[i]['annotations'])
        break
