import numpy as np
import math

class Line:
    def __init__(self, a, b):
        """Two endpoints in the form of [x, y]"""
        self.a = list(a)
        self.b = list(b)

    def is_vertical(self):
        if self.a[0] == self.b[0]:
            return True
            
        return abs((self.a[1] - self.b[1]) / (self.a[0] - self.b[0])) > 1

    def to_polyline(self):
        return np.array([
            self.a,
            self.b
        ]).astype(dtype=np.int32)

    @property
    def com(self):
        """Center of mass [x, y]"""
        return [(self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2]

    @property
    def length(self):
        """Line length"""
        return ((self.a[0] - self.b[0])**2 + (self.a[1] - self.b[1])**2) ** 0.5

    @property
    def angle(self):
        return math.atan2((self.b[1] - self.a[1]), (self.b[0] - self.a[0]))

    def __str__(self):
        return "Line([%d, %d], [%d, %d])" % \
            (self.a[0], self.a[1], self.b[0], self.b[1])

    def __repr__(self):
        return str(self)

    def intersect(self, that):
        """Intersection point of two lines"""
        denom = (self.a[0]-self.b[0])*(that.a[1]-that.b[1]) - (self.a[1]-self.b[1])*(that.a[0]-that.b[0])
        if denom == 0:
            print("Lines are parallel!")
            return [0, 0]
        return [
            ((self.a[0]*self.b[1]-self.a[1]*self.b[0])*(that.a[0]-that.b[0]) - (self.a[0]-self.b[0])*(that.a[0]*that.b[1]-that.a[1]*that.b[0])) / denom,
            ((self.a[0]*self.b[1]-self.a[1]*self.b[0])*(that.a[1]-that.b[1]) - (self.a[1]-self.b[1])*(that.a[0]*that.b[1]-that.a[1]*that.b[0])) / denom
        ]
