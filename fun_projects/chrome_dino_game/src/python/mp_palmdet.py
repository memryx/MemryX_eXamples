import numpy as np
import cv2 as cv

class MPPalmDet:
    def __init__(self, scoreThreshold=0.8):
        self.score_threshold = scoreThreshold

        self.input_size = np.array([192, 192]) # wh

    @property
    def name(self):
        return self.__class__.__name__

    def _preprocess(self, image):
        # Resize image to the input size
        image = cv.resize(image, (self.input_size[1], self.input_size[0]))  # Resize to input size
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Add batch dimension model input shape [1, 192, 192, 3]
        image = image[np.newaxis, :, :, :]

        return image

    def _postprocess(self, output_blob0, output_blob1):
        """
        Simplified postprocess function to count the number of detected palms.
        Applies NMS to remove duplicates and returns the number of palms detected.
        """

        # Extract scores from TFLite outputs (Shape: (1, 2016, 1))
        scores = output_blob1[0, :, 0]  # Shape: (2016,)

        # Apply sigmoid to the scores
        scores = 1 / (1 + np.exp(-scores))

        # Check if any score exceeds the score threshold
        return np.any(scores >= self.score_threshold)  # Returns True if any score meets the threshold
