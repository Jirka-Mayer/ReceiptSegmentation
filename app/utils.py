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
    def bhattacharyya_distance(d1, d2):
        """Calculates the bhattacharyya distance between two distributions"""
        m1, s1 = d1
        m2, s2 = d2

        # helper values
        avg_s = (s1 + s2) / 2
        avg_s_inv = np.linalg.inv(avg_s)
        detmul = np.linalg.det(s1) * np.linalg.det(s2)

        # second part that tends to the manahalobis distance
        second_part = 0.125 * (m2 - m1).T.dot(avg_s_inv.dot((m2 - m1)))

        # HACK: helps increase receipt contrast
        second_part *= 2

        # variance is really small, use only the second part of the distance
        if detmul <= 0:
            return second_part

        # first part conciders distribution variance
        first_part = 0.5 * np.log(np.linalg.det(avg_s) / np.sqrt(detmul))

        return first_part + second_part
