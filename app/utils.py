import numpy as np

class Utils:
    @staticmethod
    def calculate_distribution(pixels_luv):
        """Calculates pixel distribution in the Luv space of a given image"""
        mean = pixels_luv.mean(axis=0) # 3D vector
        variance = np.cov(pixels_luv, rowvar=False) # 3x3 matrix
        return mean, variance # 3D gaussian

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
