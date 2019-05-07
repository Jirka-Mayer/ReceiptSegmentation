from app.output import Output
import numpy as np
import cv2

class Debug:
    @staticmethod
    def show_labels(dataset_item):
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

        Output.write_image("labeled_" + dataset_item.file, img)
