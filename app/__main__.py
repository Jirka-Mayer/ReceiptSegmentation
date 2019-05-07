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
    #if item.file != "001.jpg":
    #    continue

    print(item.file)
    Debug.show_labels(item)

    # distances

    img = segmenter.normalize(item.img)
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

    # via normal ROI

    distances = segmenter.calculate_distance_map(img_luv, item.distribution())
    distances_img = segmenter.distance_map_to_img(distances, img_luv)
    Output.write_image(item.file + "_1_roi.jpg", distances_img)

    # MSER sketch

    img_mser = np.copy(distances_img)

    delta = 5
    min_area = 60
    max_area = 14400
    max_variation = 0.05
    mser = cv2.MSER_create(delta, min_area, max_area, max_variation)

    regions, bounding_boxes = mser.detectRegions(distances)
    regions = [r * segmenter.stride for r in regions]

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img_mser, hulls, 2, (0, 0, 255))

    Output.write_image(item.file + "_2_mser.jpg", img_mser)
    
    i = 0
    for r in regions:
        img_mser = np.copy(distances_img)
        
        #cv2.polylines(img_mser, [r], 2, (0, 0, 255))
        for p in r:
            cv2.circle(img_mser, tuple(p), segmenter.stride, (0, 0, 255), 2)

        Output.write_image(item.file + "_2_mser" + str(i) + ".jpg", img_mser)
        i += 1
