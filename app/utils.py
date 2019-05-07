import numpy as np

class Utils:
    @staticmethod
    def calculate_distribution(img_luv):
        """Calculates pixel distribution in the Luv space of a given image"""
        pixels_luv = img_luv.reshape(-1, 3) # flatten into a row of pixels
        mean = pixels_luv.mean(axis=0) # 3D vector
        covariance = np.cov(pixels_luv, rowvar=False) # 3x3 matrix (symmetric btw.)
        return mean, covariance # 3D gaussian

    @staticmethod
    def bhattacharyya_distance(a, b):
        """Calculates the bhattacharyya distance between two distributions"""
        m1 = a[0]
        m2 = b[0]
        s1 = a[1]
        s2 = b[1]

        avg_s = (s1 + s2) / 2
        avg_inv = np.linalg.inv(avg_s)

        # failsafe
        if np.sqrt(np.linalg.det(s1) * np.linalg.det(s2)) == 0:
            return 0

        beta = 0
        beta += 0.5 * np.log(np.linalg.det(avg_s) / np.sqrt(np.linalg.det(s1) * np.linalg.det(s2)))
        beta += 0.125 * (m2 - m1).T.dot(avg_inv.dot((m2 - m1)))

        return beta
