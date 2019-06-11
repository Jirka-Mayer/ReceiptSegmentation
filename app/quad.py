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
        ]).astype(dtype=np.int32)

    def scale(self, factor):
        return Quad(
            [self.a[0] * factor, self.a[1] * factor],
            [self.b[0] * factor, self.b[1] * factor],
            [self.c[0] * factor, self.c[1] * factor],
            [self.d[0] * factor, self.d[1] * factor]
        )
