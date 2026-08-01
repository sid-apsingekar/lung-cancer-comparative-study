import cv2
import numpy as np


class ImagePreprocessor:
    """
    Image preprocessing pipeline based on the research paper.

    Pipeline:
    1. Median Filter
    2. Histogram Equalization
    3. CLAHE
    4. Morphological Operations
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        self.kernel = np.ones((3, 3), np.uint8)

    def median_filter(self, image):
        return cv2.medianBlur(image, 3)

    def histogram_equalization(self, image):
        return cv2.equalizeHist(image)

    def apply_clahe(self, image):
        return self.clahe.apply(image)

    def morphological_operations(self, image):

        dilated = cv2.dilate(image, self.kernel, iterations=1)

        eroded = cv2.erode(dilated, self.kernel, iterations=1)

        return eroded

    def preprocess(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = self.median_filter(gray)

        gray = self.histogram_equalization(gray)

        gray = self.apply_clahe(gray)

        gray = self.morphological_operations(gray)

        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return image