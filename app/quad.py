import numpy as np

class Quad:
    def __init__(self, a, b, c, d):
        """Corners clockwise, starting from top left as lists [x, y]"""
        self.a = list(a)
        self.b = list(b)
        self.c = list(c)
        self.d = list(d)

    @staticmethod
    def from_json(json_array):
        return Quad(
            json_array[0],
            json_array[1],
            json_array[2],
            json_array[3]
        )

    def to_polyline(self):
        return np.array([
            self.a, self.b, self.c, self.d
        ])
