from app.quad import Quad
import numpy as np
import cv2

class Segmenter:
    def __init__(self):
        self.normalized_width = 800
    
    def segment(self, img):
        """Returns a quad or None if segmentation failed"""
        pass

    def normalize(self, img):
        """Returns the image with the normalized widht"""
        target_width = self.normalized_width
        scale_factor = target_width / img.shape[1]
        target_height = int(img.shape[0] * scale_factor)
        return cv2.resize(img, (target_width, target_height))
