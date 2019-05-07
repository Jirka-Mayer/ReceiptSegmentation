from app.dataset import Dataset
from app.output import Output
import cv2
import numpy as np

dataset = Dataset()
dataset.load()

Output.clear()
for item in dataset.items:
    print(item.file)

    img = np.copy(item.img)
    polylines = [item.quad.to_polyline()]
    cv2.polylines(img, polylines, True, (0, 0, 255), 5)
    
    Output.write_image(item.file, img)
