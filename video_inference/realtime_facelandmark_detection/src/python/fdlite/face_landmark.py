# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
import numpy as np
import cv2 as cv
from PIL import Image
from typing import List, Optional, Sequence, Union
from fdlite.render import Annotation
from fdlite.render import Color, landmarks_to_render_data
from fdlite.transform import image_to_tensor, sigmoid
from fdlite.transform import project_landmarks
from fdlite.types import Landmark, Rect

"""Model for face landmark detection.

Ported from Google® MediaPipe (https://google.github.io/mediapipe/).

Model card:

    https://mediapipe.page.link/facemesh-mc

Reference:

    Real-time Facial Surface Geometry from Monocular
    Video on Mobile GPUs, CVPR Workshop on Computer
    Vision for Augmented and Virtual Reality, Long Beach,
    CA, USA, 2019
"""

NUM_DIMS = 3                    # x, y, z
NUM_LANDMARKS = 468             # number of points in the face mesh
DETECTION_THRESHOLD = 0.5       # minimum score for detected faces

# face landmark connections
# (from face_landmarks_to_render_data_calculator.cc)
FACE_LANDMARK_CONNECTIONS = [
    # Lips.
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269),
    (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178),
    (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye.
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
    # Left eyebrow.
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
    # Right eye.
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
    # Right eyebrow.
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
    # Face oval.
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
MAX_FACE_LANDMARK = len(FACE_LANDMARK_CONNECTIONS)

class FaceLandmarkMXA:
    def __init__(
        self
    ) -> None:
        self.input_shape = (1,192,192,3)

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        roi: Optional[Rect] = None
    ) -> List[Landmark]:
        input_data, padding = self._preprocess(frame, roi)
        return self._postprocess(outputs, padding, image.size, roi, sync_accl=True)
        
    def _preprocess(self, frame, roi):
        height, width = self.input_shape[1:3]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=False,
            output_range=(0., 1.))
        input_data = image_data.tensor_data[np.newaxis]
        #input_data = np.transpose(input_data, axes = (1,2,0,3))
        padding = image_data.padding
        return input_data, padding

    def _postprocess(self, outputs, padding, image_size, roi):
        raw_data = np.expand_dims(outputs[1], 0)
        raw_face = np.expand_dims(outputs[0], 0)

        # second tensor contains confidence score for a face detection
        face_flag = sigmoid(raw_face).flatten()[-1]
        '''
        # no data if no face was detected
        if face_flag <= DETECTION_THRESHOLD:
            return 
        '''
        # extract and normalise landmark data
        height, width = self.input_shape[1:3]
        return project_landmarks(raw_data,
                                 tensor_size=(width, height),
                                 image_size=image_size,
                                 padding=padding,
                                 roi=roi)

def face_landmarks_to_render_data(
    face_landmarks: Sequence[Landmark],
    landmark_color: Color,
    connection_color: Color,
    thickness: float = 2.0,
    output: Optional[List[Annotation]] = None
) -> List[Annotation]:
    """Convert face landmarks to render data.

    This post-processing function can be used to generate a list of rendering
    instructions from face landmark detection results.

    Args:
        face_landmarks (list): List of `Landmark` detection results returned
            by `FaceLandmark`.

        landmark_color (Color): Color of the individual landmark points.

        connection_color (Color): Color of the landmark connections that
            will be rendered as lines.

        thickness (float): Width of the lines and landmark point size in
            viewport units (e.g. pixels).

        output (list): Optional list of render annotations to add the items
            to. If not provided, a new list will be created.
            Use this to add multiple landmark detections into a single render
            annotation list.

    Returns:
        (list) List of render annotations that should be rendered.
        All positions are normalized, e.g. with a value range of [0, 1].
    """
    render_data = landmarks_to_render_data(
        face_landmarks, FACE_LANDMARK_CONNECTIONS,
        landmark_color, connection_color, thickness,
        normalized_positions=True, output=output)
    return render_data
