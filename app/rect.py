import numpy as np

class Rect:
    def __init__(self, a, b):
        """Upper left corner and then the opposite one as lists [x, y]"""
        self.a = list(a)
        self.b = list(b)

    @staticmethod
    def from_json(json_array):
        return Rect(
            json_array[0],
            json_array[1]
        )

    def floor(self):
        return Rect(
            map(int, self.a),
            map(int, self.b)
        )

    def to_polyline(self):
        return np.array([
            self.a,
            [self.b[0], self.a[1]],
            self.b,
            [self.a[0], self.b[1]]
        ])
