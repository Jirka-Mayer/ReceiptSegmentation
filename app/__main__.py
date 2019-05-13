from app.dataset import Dataset
from app.output import Output
from app.segmenter import Segmenter
from app.utils import Utils
from app.debug import Debug
import numpy as np
import cv2

dataset = Dataset()
dataset.load()

segmenter = Segmenter()

Output.clear()
for item in dataset.items:
    print(item.file)
    
    # run the segmentation algorithm

    segmenter.segment(item.img, item.distribution())

    # draw the result

    if segmenter.region is None:
        print("No region found.")
        continue

    hull = cv2.convexHull(segmenter.region)[:,0,:]
    img = segmenter.draw_region_over(segmenter.img_normalized, segmenter.region)
    cv2.polylines(img, [hull], True, (255, 0, 255), 5)

    #Debug.show_labels(item)
    Output.write_image(item.file + "_1_result.jpg", img)
