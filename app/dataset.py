from app.quad import Quad
from app.rect import Rect
from app.utils import Utils
import json
import cv2

class DatasetItem:
    def __init__(self, dataset, json_object):
        self.file = json_object["img"]
        self.img = cv2.imread(dataset.folder + "/img/" + self.file) # BGR format
        
        self.difficulty = json_object["difficulty"]
        
        self.quad = Quad.from_json(json_object["quad"])

        self.distribution_rect = Rect.from_json(json_object["distribution"])
        r = self.distribution_rect.floor()
        self.distribution_img = self.img[r.a[1]:r.b[1], r.a[0]:r.b[0]]

    def distribution(self):
        """Calculates the distribution of the selected region of interest"""
        img_luv = cv2.cvtColor(self.distribution_img, cv2.COLOR_BGR2Luv)
        return Utils.calculate_distribution(img_luv)

    def small_roi_rect(self):
        """Calculates the rectangle of the small ROI inside the main ROI"""
        center_x = (self.distribution_rect.a[0] + self.distribution_rect.b[0]) // 2
        center_y = (self.distribution_rect.a[1] + self.distribution_rect.b[1]) // 2
        semi_size = 50 // 2
        return Rect(
            [center_x - semi_size, center_y - semi_size],
            [center_x + semi_size, center_y + semi_size]
        )

    def small_distribution(self):
        """Calculates distribution of the small ROI"""
        r = self.small_roi_rect().floor()
        img = self.img[r.a[1]:r.b[1], r.a[0]:r.b[0]]
        img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
        return Utils.calculate_distribution(img_luv)

class Dataset:
    def __init__(self, folder="dataset"):
        self.folder = folder
        self.items = []

    def load(self):
        info = json.load(open(self.folder + "/dataset.json", "r"))
        self.items = [DatasetItem(self, i) for i in info]
