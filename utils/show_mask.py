from pathlib import Path
import pandas as pd    
import cv2
import numpy as np 


def show_label_mask(ROOT , id , annotations , source_wsi , dataset):
    # print(str(ROOT / "train" / f"{id}.tif"))
    frame = cv2.imread(str(ROOT / "train" / f"{id}.tif"))

    for i in range(len(annotations)):
        pts = np.array(annotations[i]['coordinates'])
        color = (255,0,0)
        if annotations[i]['type'] == 'glomerulus':
            color = (0,255,0)
        elif annotations[i]['type'] == 'unsure':
            color = (0,0,255)
        # else if annotations[i]['type'] == 'glomerulus':
        cv2.polylines(frame , pts , True , color , 3)
        # cv2.fillPoly(frame, pts, color)
    
    cv2.putText(frame , f"WSI {source_wsi} DS {dataset}" , (0,25) , cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("frame" , frame)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    ROOT = Path("hubmap-hacking-the-human-vasculature")
    labels = pd.read_json(ROOT / "polygons.jsonl" , lines=True)
    matadata = pd.read_csv(ROOT / "tile_meta.csv")
    for i in range(len(labels)):
        source_wsi , dataset = matadata[matadata["id"] == labels.iloc[i]['id']][["source_wsi" , "dataset"]].values[0]
        show_label_mask(ROOT , labels.iloc[i]['id'] , labels.iloc[i]['annotations'] ,source_wsi , dataset)


