from app.quad import Quad
from app.line import Line
from app.utils import Utils
import numpy as np
import cv2
import math

class Segmenter:
    def __init__(self):
        # normalization
        self.normalized_width = 800
        
        # distance map
        self.window_size = 10 # px
        self.stride = 10 # px

        # intermediates for debugging
        self.img_original = None
        self.img_normalized = None
        self.img_preprocessed = None
        self.distances_img = None
        self.region = None # in normalized img coordinates (list of pixel coords)
        self.img_quad = None
        self.quad = None # in normalized img coordinates
    
    def segment(self, img, target_distribution):
        """Returns a quad or None if segmentation failed"""
        self.img_original = img

        img = self.normalize(img)
        img = self.preprocess(img)
        img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
        distances = self.calculate_distance_map(img_luv, target_distribution)
        region = self.get_receipt_pixels(distances)

        if region is None:
            return None

        quad = self.extract_quad(region)
        return quad

    def normalize(self, img):
        """Returns the image with the normalized width"""
        target_width = self.normalized_width
        scale_factor = target_width / img.shape[1]
        target_height = int(img.shape[0] * scale_factor)
        
        # DEBUG
        self.img_normalized = cv2.resize(img, (target_width, target_height))
        
        return self.img_normalized

    def preprocess(self, img):
        """Does some preprocessing to increase accuracy"""

        """img = cv2.bilateralFilter(img, 9, 75, 75)

        alpha = 2.0
        beta = 0.0
        img = np.clip(alpha*img + beta, 0, 255).astype(dtype=np.uint8)"""

        # DEBUG
        self.img_preprocessed = img
        
        return self.img_preprocessed

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

        # DEBUG
        self.distances_img = self.distance_map_to_img(distances, img_luv)

        return distances

    def distance_map_to_img(self, distances, img):
        """Resizes the distance map back up to fit the image size"""
        distances_img = cv2.resize(distances, (img.shape[1], img.shape[0]))
        return np.dstack([distances_img, distances_img, distances_img])

    def get_receipt_pixels(self, distances):
        """Returns a list of points (pixel positions)
            that are the receipt or None"""
        delta = 5
        min_area = 60
        max_area = 14400
        max_variation = 0.05
        mser = cv2.MSER_create(delta, min_area, max_area, max_variation)

        regions, bounding_boxes = mser.detectRegions(distances)
        regions = [r * self.stride for r in regions]

        if len(regions) == 0:
            self.region = None
            return None

        # DEBUG
        self.region = regions[0]
        
        return regions[0]

    def draw_region_over(self, img, region):
        """Draws a region over a given image"""
        img = np.copy(img)
        for p in region:
            cv2.circle(img, tuple(p), self.stride // 2, (0, 0, 255), 1)
        return img

    def extract_quad(self, region):
        """Extracts a quadrangle from a region"""

        # convex hull
        hull = cv2.convexHull(region)[:,0,:]

        # lines
        lines = [Line(a, b) for (a, b) in zip(hull, list(hull[1:]) + [hull[0]])]
        verticals = [l for l in lines if l.is_vertical()]
        horizontals = [l for l in lines if not l.is_vertical()]

        # center of mass [x, y]
        com = list(region.sum(axis=0) // region.shape[0])

        # extremes
        tops = [l for l in horizontals if l.com[1] < com[1]]
        bottoms = [l for l in horizontals if l.com[1] >= com[1]]
        lefts = [l for l in verticals if l.com[0] < com[0]]
        rights = [l for l in verticals if l.com[0] >= com[0]]

        # quad not found
        if tops == [] or bottoms == [] or lefts == [] or rights == []:
            self.quad = None
            return None

        def sgn(x):
            """Sign of x"""
            if x == 0:
                return 0
            return x / abs(x)

        def find_average(source_lines, vertical):
            """Computes the average line of a set of lines"""
            coms = np.array([l.com for l in source_lines])
            angles = np.array([
                l.angle + math.pi if l.angle < 0 else l.angle
                for l in source_lines
            ]) if vertical else np.array([
                l.angle - sgn(l.angle) * math.pi if abs(l.angle) > math.pi/2 else l.angle
                for l in source_lines
            ])
            weights = np.array([l.length for l in source_lines])

            com = (coms * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
            angle = (angles * weights).sum(axis=0) / weights.sum()

            d = np.array([math.cos(angle), math.sin(angle)]) * 1000
            return Line(com - d, com + d)

        # average out
        top = find_average(tops, False)
        bottom = find_average(bottoms, False)
        left = find_average(lefts, True)
        right = find_average(rights, True)

        # intersections
        self.quad = Quad(
            top.intersect(left),
            top.intersect(right),
            bottom.intersect(right),
            bottom.intersect(left)
        )

        # DEBUG DRAW
        img = self.img_normalized.copy()
        img = cv2.polylines(img, [l.to_polyline() for l in tops], False, (255, 0, 0), 5)
        img = cv2.polylines(img, [l.to_polyline() for l in bottoms], False, (255, 255, 0), 5)
        img = cv2.polylines(img, [l.to_polyline() for l in lefts], False, (0, 0, 255), 5)
        img = cv2.polylines(img, [l.to_polyline() for l in rights], False, (0, 255, 255), 5)
        img = cv2.polylines(img, [
            l.to_polyline() for l in [top, bottom, left, right]
        ], False, (0, 255, 0), 2)
        img = cv2.polylines(img, [
            self.quad.to_polyline()
        ], True, (255, 0, 255), 2)
        self.img_quad = img
        """import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()"""
        # DEBUG DRAW END

        return self.quad.scale(self.img_original.shape[1] / self.normalized_width)
