import numpy as np
import os, cv2
from queue import Queue, Empty
from dataclasses import dataclass, field
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

from memryx import AsyncAccl

@dataclass
class HandPose():
    """
    Results for a single detected hand!

    bbox                       : hand bounding box found in image of format [x1, y1, x2, y2] (top-left and bottom-right points)
    landmarks                  : screen landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], z value is relative to WRIST
    rotated_landmarks_world    : world landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], 3D metric x, y, z coordinate
    handedness                 : str (Left or Right)
    confidence                 : confidence
    """

    bbox: np.ndarray                     = field(default_factory=lambda: [])
    landmarks: np.ndarray                = field(default_factory=lambda: [])
    rotated_landmarks_world: np.ndarray  = field(default_factory=lambda: [])
    handedness: str   = 'None'
    confidence: float = 0.0

@dataclass
class AnnotatedFrame():
    image: np.ndarray
    num_detections: int = 0
    handposes: list[HandPose] = field(default_factory=lambda: [])


class MxHandPose:
    def __init__(self, mx_modeldir, num_hands, **kwargs):

        self._stopped            = False
        self._outstanding_frames = 0

        # queues                                   # dimensions
        self.input_q           = Queue(maxsize=2)  #(annotated_frame)
        self.stage0_q          = Queue(maxsize=2)  #(annotated_frame, pad_bias)
        self.stage1_q          = Queue(maxsize=3)  #(annotated_frame) --> has detected palm info!
        self.stage2_q          = Queue(maxsize=3)  #(annotated_frame, rotated_palm_bbox, angle, rotation_matrix, pad_bias)
        self.output_q          = Queue(maxsize=4)  #(annotated_frame with results)


        # models
        self.num_hands         = num_hands
        self.palmdet_model     = MPPalmDet(topK=self.num_hands)
        self.handpose_model    = MPHandPose(confThreshold=0.5)

        dfp_path               = os.path.join(mx_modeldir, 'models.dfp')

        # Initialize the accelerator with the model
        self.accl = AsyncAccl(dfp_path, group_id=0)

        # Connect input and output functions to the accelerator
        self.accl.connect_input(self._palmdetect_src, model_idx=1)
        self.accl.connect_output(self._palmdetect_sink, model_idx=1)
        self.accl.connect_input(self._handpose_src, model_idx=0)
        self.accl.connect_output(self._handpose_sink, model_idx=0)

    # top-level input (images in)
    def put(self, image, block=True, timeout=None):
        annotated_frame = AnnotatedFrame(np.array(image))
        self.input_q.put(annotated_frame, block, timeout)
        self._outstanding_frames += 1

    # final outputs (hand landmark data out)
    def get(self, block=True, timeout=None):
        self._outstanding_frames -= 1
        annotated_frame = self.output_q.get(block, timeout)
        self.output_q.task_done()
        return annotated_frame

    def __del__(self):
        if not self._stopped:
            self.stop()

    def stop(self):
        while self._outstanding_frames > 0:
            try:
                self.get(timeout=0.1)
            except Empty:
                continue

        self.input_q.put(None)
        self.stage1_q.put(None)
        self._stopped = True

    # is not currently running anything
    def empty(self):
        return self.output_q.empty() and self.input_q.empty()

    # can't accept more input images
    def full(self):
        return self.input_q.full()

    #####################################################################################################
    # Async Functions
    #####################################################################################################

    # palm detect input callback & preproc
    def _palmdetect_src(self):
        annotated_frame = self.input_q.get()
        if annotated_frame is None:
            return None
        self.input_q.task_done()

        annotated_frame.image = cv2.flip(annotated_frame.image,1)

        ifmap, pad_bias= self.palmdet_model._preprocess(annotated_frame.image)
        self.stage0_q.put((annotated_frame, pad_bias))

        ifmap = np.squeeze(ifmap, 0)
        ifmap = np.expand_dims(ifmap, 2)

        return ifmap

    # palm detect output callback & post-proc
    def _palmdetect_sink(self, *accl_outputs):

        # this section implments the cropped TFLite post model
        # using only numpy functions, to avoid having to spin
        # up a session (it's also faster)
        ###########################################################

        # m0: [12, 12, 6]    --> out0
        # m1: [12, 12, 108]  ===> out1
        # m2: [24, 24, 2]    --> out0
        # m3: [24, 24, 36]   ===> out1

        # out0: [1,2016,1]
        # out1: [1,2016,18]

        m0 = accl_outputs[0]
        m1 = accl_outputs[1]
        m2 = accl_outputs[2]
        m3 = accl_outputs[3]

        r0a = np.reshape(m2, (1,1152,1))
        r0b = np.reshape(m0, (1,864,1))
        r0  = np.concatenate((r0a, r0b), axis=1)

        r1a = np.reshape(m3, (1,1152,18))
        r1b = np.reshape(m1, (1,864,18))
        r1  = np.concatenate((r1a, r1b), axis=1)

        # end post-model
        ###########################################################

        annotated_frame, pad_bias = self.stage0_q.get()
        self.stage0_q.task_done()
        h, w, _ = annotated_frame.image.shape
        palms    = self.palmdet_model._postprocess([r0, r1],  np.array([w, h]), pad_bias)

        # Count number of detected hands
        annotated_frame.num_detections = len(palms)

        if annotated_frame.num_detections == 0: # no hands have been detected!
            self.output_q.put(annotated_frame)
            return

        for palm in palms:
            self.stage1_q.put((annotated_frame, palm))

    # hand model input callback & preproc
    def _handpose_src(self):

        data = self.stage1_q.get()

        if data is None:
            return None

        self.stage1_q.task_done()

        annotated_frame, palm = data

        ifmap, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self.handpose_model._preprocess(annotated_frame.image, palm)
        self.stage2_q.put((annotated_frame, rotated_palm_bbox, angle, rotation_matrix, pad_bias))
        ifmap = np.squeeze(ifmap, 0)
        ifmap = np.expand_dims(ifmap, 2)

        return ifmap

    # hand model output callback & postproc
    def _handpose_sink(self, *accl_outputs):

        annotated_frame, rotated_palm_bbox, angle, rotation_matrix, pad_bias  = self.stage2_q.get()
        self.stage2_q.task_done()
        handpose = self.handpose_model._postprocess(accl_outputs, rotated_palm_bbox, angle, rotation_matrix, pad_bias)

        if handpose is None:
            self.output_q.put(annotated_frame)
            return


        bbox                    = handpose['bbox']
        landmarks               = handpose['landmarks']
        rotated_landmarks_world = handpose['rotated_landmarks_world']
        handedness              = handpose['handedness']
        confidence              = handpose['conf']

        annotated_frame.handposes.append(HandPose(bbox, landmarks, rotated_landmarks_world, handedness, confidence))
        if len(annotated_frame.handposes) == annotated_frame.num_detections:
            self.output_q.put(annotated_frame)



