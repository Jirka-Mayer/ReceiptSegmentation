from app.dataset import Dataset
from app.output import Output
from app.segmenter import Segmenter
from app.utils import Utils
from app.segmenter import Segmenter
import numpy as np
import cv2

dataset = Dataset()
dataset.load()

segmenter = Segmenter()

Output.clear()
for item in dataset.items:
    if item.file != "006.jpg":
        continue

    print(item.file)

    # labels
    img = np.copy(item.img)
    polylines = [
        item.quad.to_polyline(),
        item.distribution_rect.to_polyline()
    ]
    cv2.polylines(img, polylines, True, (0, 0, 255), 5)
    Output.write_image(item.file + "_labels.jpg", img)

    # ----------------------

    kernel = 10

    distribution_img_luv = cv2.cvtColor(item.distribution_img, cv2.COLOR_BGR2Luv)
    distribution = Utils.calculate_distribution(distribution_img_luv)

    img = segmenter.normalize(item.img)

    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

    distances = np.empty(
        dtype=np.uint8,
        shape=(int(img_luv.shape[0] / kernel), int(img_luv.shape[1] / kernel))
    )

    for x in range(distances.shape[1]):
        print(x, "/", distances.shape[1])
        for y in range(distances.shape[0]):
            distr = Utils.calculate_distribution(img_luv[y*kernel:(y+1)*kernel, x*kernel:(x+1)*kernel])
            distance = Utils.bhattacharyya_distance(distr, distribution)
            if distance > 255: distance = 255
            if distance < 0: distance = 0
            if np.isnan(distance): distance = 0
            distances[y, x] = int(distance)

    Output.write_image(item.file, distances)
