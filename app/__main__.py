from app.dataset import Dataset
from app.output import Output
from app.segmenter import Segmenter
from app.utils import Utils
from app.segmenter import Segmenter
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

    # ----------------------

    distribution_img_luv = cv2.cvtColor(item.distribution_img, cv2.COLOR_BGR2Luv)
    distribution = Utils.calculate_distribution(distribution_img_luv)

    img = segmenter.normalize(item.img)
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

    distances = segmenter.calculate_distance_map(img_luv, distribution)
    distances_img = cv2.resize(distances, (img.shape[1], img.shape[0]))
    distances_img = np.dstack([distances_img, distances_img, distances_img])

    Output.write_image(item.file, distances_img)

    combined = img - distances_img
    combined[img < distances_img] = 0
    Output.write_image("combined_" + item.file, combined)
