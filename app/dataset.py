from app.quad import Quad
from app.rect import Rect
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

class Dataset:
    def __init__(self, folder="dataset"):
        self.folder = folder
        self.items = []

    def load(self):
        info = json.load(open(self.folder + "/dataset.json", "r"))
        self.items = [DatasetItem(self, i) for i in info]
