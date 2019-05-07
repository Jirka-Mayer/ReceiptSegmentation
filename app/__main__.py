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
    #if item.file != "008.jpg":
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

    #combined = img - distances_img
    #combined[img < distances_img] = 0
    #Output.write_image(item.file + "_3_combined.jpg", combined)

    # via small ROI

    distances = segmenter.calculate_distance_map(img_luv, item.small_distribution())
    distances_img = segmenter.distance_map_to_img(distances, img_luv)
    Output.write_image(item.file + "_2_small_roi.jpg", distances_img)
