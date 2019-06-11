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
    
    #if item.file != "005.jpg":
    #    continue
    
    #
    # run the segmentation algorithm
    #

    quad = segmenter.segment(item.img, item.distribution())

    #
    # draw the result
    # 

    # reference image
    Output.write_image(item.file + "_0_norm.jpg", segmenter.img_normalized)

    # distance map
    Output.write_image(item.file + "_1_distances.jpg", segmenter.distances_img)

    if segmenter.region is None:
        print("No region found.")
        continue
    
    # poly to quad process
    Output.write_image(item.file + "_2_quad.jpg", segmenter.img_quad)

    # image with resulting quad only
    Output.write_image(
        item.file + "_3_result.jpg",
        cv2.polylines(item.img, [quad.to_polyline()], True, (255, 0, 0), 10)
    )
