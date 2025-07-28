import os
import queue
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import torch
from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox
import memryx as mx

@dataclass
class NMSArgs:
    conf_thres: float = 0.25
    iou_thres: float = 0.7
    max_det: int = 300
    nc: int = 15
    classes = None
    rotated: bool = True

@dataclass
class Detection:
    bbox: list[int]
    rot: float
    conf: float
    cls: str

@dataclass
class AnnotatedFrame():
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)

class MXObb:
    detector_imgsz = 1024
    letterbox = LetterBox((detector_imgsz, detector_imgsz))
    classes =  {
        0: 'plane', 
        1: 'ship', 
        2: 'storage tank', 
        3: 'baseball diamond', 
        4: 'tennis court', 
        5: 'basketball court', 
        6: 'ground track field', 
        7: 'harbor', 
        8: 'bridge', 
        9: 'large vehicle', 
        10: 'small vehicle', 
        11: 'helicopter', 
        12: 'roundabout', 
        13: 'soccer ball field', 
        14: 'swimming pool'
    }

    def __init__(self, models_dir, nms_args = None):
        if nms_args is None:
            self.nms_args = NMSArgs()
        else:
            self.nms_args = nms_args

        self._stopped = False
        self._outstanding_frames = 0

        self.input_q  = queue.Queue(maxsize=1)
        self.stage0_q = queue.Queue(maxsize=2)
        self.output_q = queue.Queue(maxsize=1)


        self.accl = mx.AsyncAccl(str(Path(models_dir) / "yolov8s-obb.dfp"))
        self.accl.set_postprocessing_model(str(Path(models_dir) / "yolov8s-obb_post.onnx"))
        self.accl.connect_input(self._detector_source)
        self.accl.connect_output(self._detector_sink)

    def put(self, image, block=True, timeout=None):
        annotated_frame = AnnotatedFrame(np.array(image))
        self.input_q.put(annotated_frame, block, timeout)
        self._outstanding_frames += 1

    def get(self, block=True, timeout=None):
        self._outstanding_frames -= 1
        annotated_frame = self.output_q.get(block, timeout)
        return annotated_frame

    def empty(self):
        return self.output_q.empty() and self.input_q.empty()  

    def full(self):
        return self.input_q.full()

    ### Async Functions #######################################################
    def _detector_source(self):
        annotated_frame = self.input_q.get()
        
        if annotated_frame is None:
            return None

        self.stage0_q.put(annotated_frame)

        ifmap = self._preprocess(annotated_frame.image)
        ifmap = np.expand_dims(ifmap, 0)
        ifmap = np.transpose(ifmap, (0,3,1,2))
        return ifmap.astype(np.float32)

    def _detector_sink(self, *outputs):
        annotated_frame = self.stage0_q.get()

        image = annotated_frame.image
        #annotated_frame.detections = self._postprocess(outputs[0], image.shape)
        detections = self._postprocess(outputs[0], image.shape)
        for det in detections:
            annotated_frame.detections.append(
                Detection(
                    bbox = det[:4],
                    rot = det[4],
                    conf = det[5],
                    cls = self.classes[det[6]]
                )
            )

        self.output_q.put(annotated_frame)

    # Pre / Post Processing app
    def _preprocess(self, image):
        resized = self.letterbox(image=image)
        normalized = resized / 255.0
        return normalized.astype(np.float32)
    
    #def _postprocess(self, outputs, network_shape, original_shape):
    def _postprocess(self, outputs, original_shape):
        outputs = torch.Tensor(outputs)
        #pred = ops.non_max_suppression(outputs, **args)[0]
        pred = ops.non_max_suppression(outputs, **self.nms_args.__dict__)[0]
    
        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        network_shape = (self.detector_imgsz, self.detector_imgsz)
        rboxes[:, :4] = ops.scale_boxes(network_shape, rboxes[:, :4], original_shape, xywh=True)
        # xywh, r, conf, cls
        boxes = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        return boxes.numpy()
    
    def stop(self):
        while self._outstanding_frames > 0:
            try:
                self.get(timeout=0.1)
            except queue.Empty:
                continue
            
        self.input_q.put(None)
        self._stopped = True

