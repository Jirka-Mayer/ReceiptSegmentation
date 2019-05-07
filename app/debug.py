from app.output import Output
import numpy as np
import cv2

class Debug:
    @staticmethod
    def show_labels(dataset_item):
        """Puts the dataset image into the output dir with the labeled quad"""
        img = np.copy(dataset_item.img)
        
        # red receipt outline
        cv2.polylines(
            img,
            [dataset_item.quad.to_polyline()],
            True,
            (0, 0, 255),
            5
        )

        # purple region of interest
        cv2.polylines(
            img,
            [dataset_item.distribution_rect.to_polyline()],
            True,
            (255, 0, 255),
            3
        )

        # blue inner region of interest
        cv2.polylines(
            img,
            [dataset_item.small_roi_rect().to_polyline()],
            True,
            (255, 0, 0),
            2
        )

        Output.write_image(dataset_item.file + "_0_labeled.jpg", img)

    @staticmethod
    def compare_distribution_to_histogram(dataset_item):
        """Shows histogram vs. computed distribution"""
        # LUMINOSITY ONLY
        img_luv = cv2.cvtColor(dataset_item.distribution_img, cv2.COLOR_BGR2Luv)
        pixels = img_luv.reshape(-1, 3)
        
        import matplotlib.pyplot as plt
        plt.hist(pixels[:,0], bins=50) # luminosity
        plt.hist(pixels[:,1], bins=50) # u
        plt.hist(pixels[:,2], bins=50) # v
        #plt.show()
        plt.savefig("out/" + dataset_item.file)
        plt.clf()

