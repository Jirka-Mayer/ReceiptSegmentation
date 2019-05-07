import cv2
import os
import shutil

class Output:
    @staticmethod
    def clear():
        """Clears the output directory"""
        shutil.rmtree("out")
        os.makedirs("out")

    @staticmethod
    def write_image(name, image):
        cv2.imwrite("out/" + name, image)
