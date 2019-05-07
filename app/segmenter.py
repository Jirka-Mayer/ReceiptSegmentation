from app.quad import Quad
from app.utils import Utils
import numpy as np
import cv2

class Segmenter:
    def __init__(self):
        # normalization
        self.normalized_width = 800
        
        # distance map
        self.window_size = 10 # px
        self.stride = 10 # px
    
    def segment(self, img):
        """Returns a quad or None if segmentation failed"""
        pass

    def normalize(self, img):
        """Returns the image with the normalized widht"""
        target_width = self.normalized_width
        scale_factor = target_width / img.shape[1]
        target_height = int(img.shape[0] * scale_factor)
        return cv2.resize(img, (target_width, target_height))

    def calculate_distance_map(self, img_luv, target_distribution):
        """Calculates a distance map by sliding a window by a given stride"""
        distances = np.empty(
            dtype=np.uint8,
            shape=(
                int((img_luv.shape[0] - self.window_size) / self.stride),
                int((img_luv.shape[1] - self.window_size) / self.stride)
            )
        )

        for x in range(distances.shape[1]):
            for y in range(distances.shape[0]):
                x_px = x * self.stride
                y_px = y * self.stride

                local_distribution = Utils.calculate_distribution(
                    img_luv[
                        y_px : y_px + self.window_size,
                        x_px : x_px + self.window_size
                    ]
                )
                distance = Utils.bhattacharyya_distance(
                    local_distribution,
                    target_distribution
                )
                
                # clamp & check distance
                if np.isnan(distance):
                    print("Distance calculated was NaN! Setting to 255.")
                    distance = 255
                if distance > 255: distance = 255
                if distance < 0: distance = 0

                distances[y, x] = int(distance)

        return distances

    def distance_map_to_img(self, distances, img):
        """Resizes the distance map back up to fit the image size"""
        distances_img = cv2.resize(distances, (img.shape[1], img.shape[0]))
        return np.dstack([distances_img, distances_img, distances_img])
