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
       
        # queues
        self.input_q           = Queue(maxsize=2)  #(annotated_frame)
        self.stage0_q          = Queue(maxsize=2)  #(annotated_frame, pad_bias)
        self.stage1_q          = Queue(maxsize=3)  #(annotated_frame) --> has detected palm info!
        self.stage2_q          = Queue(maxsize=3)  #(annotated_frame, rotated_palm_bbox, angle, rotation_matrix, pad_bias)
        self.output_q          = Queue(maxsize=4)  #(annotated_frame with results)


        # models
        self.num_hands         = num_hands
        self.palmdet_model     = MPPalmDet(topK=self.num_hands)
        self.handpose_model    = MPHandPose(confThreshold=0.5)

        # 
        dfp_path               = os.path.join(mx_modeldir, 'models.dfp')
        palmdet_postprocess    = os.path.join(mx_modeldir, 'model_1_palm_detection_lite_post.tflite')

        # Initialize the accelerator with the model
        self.accl = AsyncAccl(dfp_path, device_ids=0)

        # Connect pre & post
        self.accl.set_postprocessing_model(palmdet_postprocess, model_idx=1)

        # Connect input and output functions to the accelerator
        self.accl.connect_input(self._palmdetect_src, model_idx=1)
        self.accl.connect_output(self._palmdetect_sink, model_idx=1)
        self.accl.connect_input(self._handpose_src, model_idx=0)
        self.accl.connect_output(self._handpose_sink, model_idx=0)

    def put(self, image, block=True, timeout=None):
        annotated_frame = AnnotatedFrame(np.array(image))
        self.input_q.put(annotated_frame, block, timeout)      
        self._outstanding_frames += 1

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

    def empty(self):
        return self.output_q.empty() and self.input_q.empty()  

    def full(self):
        return self.input_q.full()
    
    #####################################################################################################
    # Async Functions
    #####################################################################################################

    def _palmdetect_src(self):
        annotated_frame = self.input_q.get()
        if annotated_frame is None:
            return None
        self.input_q.task_done()

        annotated_frame.image = cv2.flip(annotated_frame.image,1)

        ifmap, pad_bias= self.palmdet_model._preprocess(annotated_frame.image)
        self.stage0_q.put((annotated_frame, pad_bias))

        #ifmap = np.squeeze(ifmap, 0)
        #ifmap = np.expand_dims(ifmap, 0)

        return ifmap
    
    def _palmdetect_sink(self, *accl_outputs):
        annotated_frame, pad_bias = self.stage0_q.get()
        self.stage0_q.task_done()
        h, w, _ = annotated_frame.image.shape
        palms    = self.palmdet_model._postprocess(accl_outputs,  np.array([w, h]), pad_bias)
    
        # Count number of detected hands
        annotated_frame.num_detections = len(palms)

        if annotated_frame.num_detections == 0: # no hands have been detected!
            self.output_q.put(annotated_frame)
            return

        for palm in palms:
            self.stage1_q.put((annotated_frame, palm))

    def _handpose_src(self):

        data = self.stage1_q.get()

        if data is None:
            return None

        self.stage1_q.task_done()
        
        annotated_frame, palm = data

        ifmap, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self.handpose_model._preprocess(annotated_frame.image, palm) 
        self.stage2_q.put((annotated_frame, rotated_palm_bbox, angle, rotation_matrix, pad_bias))
        #ifmap = np.squeeze(ifmap, 0)
        #ifmap = np.expand_dims(ifmap, 2)

        return ifmap
    
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



