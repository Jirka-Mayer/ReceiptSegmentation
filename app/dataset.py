from app.quad import Quad
import json
import cv2

class DatasetItem:
    def __init__(self, dataset, json_object):
        self.file = json_object["img"]
        self.img = cv2.imread(dataset.folder + "/img/" + self.file)
        self.difficulty = json_object["difficulty"]
        self.quad = Quad.from_json(json_object["quad"])

class Dataset:
    def __init__(self, folder="dataset"):
        self.folder = folder
        self.items = []

    def load(self):
        info = json.load(open(self.folder + "/dataset.json", "r"))
        self.items = [DatasetItem(self, i) for i in info]
