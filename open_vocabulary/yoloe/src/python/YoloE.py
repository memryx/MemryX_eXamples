import cv2
import queue
import numpy as np
from pathlib import Path
import threading
from dataclasses import dataclass, field
from ultralytics.models.yolo.segment import SegmentationPredictor
from memryx.neural_compiler.graph.engine.graph import MxGraph
from memryx.utilities.partial_compile import compile_mapped
from PIL import Image
import torch
import memryx
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import time
import pickle
import onnx
from ultralytics.utils.plotting import Colors
import os, shutil


COLOR_WHEEL = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (128, 0, 128),     # Purple
    (255, 165, 0),     # Orange
    (0, 128, 128),     # Teal
    (128, 128, 0),     # Olive
]

def letterbox(frame, img_size):
    """
    Resizes and pads the given frame to fit into a square of size img_size while maintaining aspect ratio.
    
    Parameters:
        frame (numpy.ndarray): The input image as a NumPy array.
        img_size (int): The desired size of the output square image.

    Returns:
        numpy.ndarray: The letterboxed image as a NumPy array.
        tuple: Padding applied as (pad_left, pad_top, pad_right, pad_bottom).
    """
    # Get the original dimensions
    height, width = frame.shape[:2]

    # Calculate scale to maintain aspect ratio
    scale = img_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_top = (img_size - new_height) // 2
    pad_bottom = img_size - new_height - pad_top
    pad_left = (img_size - new_width) // 2
    pad_right = img_size - new_width - pad_left

    # Add padding with black color
    letterboxed = cv2.copyMakeBorder(
        resized_frame,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )
    return letterboxed, (pad_left, pad_top, pad_right, pad_bottom)

def setup_predictor(yoloe, classes=['person'], **kwargs):
    yoloe.set_classes(classes, yoloe.get_text_pe(classes))
    args = dict(model=yoloe.model, **kwargs)
    predictor = SegmentationPredictor(overrides=args)
    predictor.setup_model(None)
    predictor.imgsz = kwargs['imgsz']
    predictor.batch = [['None']]
    return predictor

@dataclass
class BBox():
    # Bounding box coords [(left,top,width,height), ...]
    cls: str
    bbox: list[tuple[int, int, int, int]]

@dataclass
class AnnotatedFrame():
    image: np.ndarray
    num_detections: int = 0
    detections : list[BBox] = field(default_factory=lambda: [])
    dets = None

@dataclass
class Framedata():
    orig_frame: np.ndarray
    do_seg: bool = True
    do_det: bool = True
    do_label: bool = True
    rm_background: bool = False
    do_blur: bool = False

placeholder = np.zeros([1,3,640,640])

class YoloE:
    _img_size = 640

    def __init__(self, version='v8',size='s', class_names=['person']):
        self._stop_lock = threading.Lock()

        self._conf = 0.25
        self._iou = 0.7  
        self._darkness = 0.3 

        self.classes = class_names 
        self.selected_classes = class_names.copy()
        self.size = size
        self.version = version

        self._assets_dir = Path(f'../../models')
        self._model_path = self._assets_dir / Path(f'yoloe-{version}{size}-seg')

        if self._assets_dir.exists() == False:
            raise ValueError(f"Assets for classes and model size='{size}' does not exsist.")

        self.yoloe = YOLO(f'yoloe-{version}{size}-seg.pt')

        self._accl = None
        self.do_seg=True
        self.do_det = True
        self.do_label=True
        self.rm_background = False
        self.do_blur = False
        self._blur_sigma = 0
        self.input_q = queue.Queue(maxsize=1)
        self.middle_q = queue.Queue(maxsize=2)
        self.output_q = queue.Queue(maxsize=1)


        self._predictor = setup_predictor(self.yoloe,
                                          imgsz=self._img_size, 
                                          conf=self._conf, 
                                          iou=self._iou,
                                          classes=self.classes)

        self._outstanding_frames = 0
        self._stopped = True 

        self.connect_accl()
    
    @property
    def darkness(self):
        return self._darkness

    @darkness.setter
    def darkness(self, val):
        if val < 0 or val > 1.0:
            print(f'Invalid darkness value {val}')
            return
        self._darkness = val
    
    @property
    def blur_sigma(self):
        return self._blur_sigma

    @blur_sigma.setter
    def blur_sigma(self, val):
        if val < 0:
            print(f'Invalid blur_sigma value {val}')
            return
        self._blur_sigma = val
    
    @property
    def selected_classes(self):
        return self._selected_classes

    @selected_classes.setter
    def selected_classes(self, classes):
        for clas in classes:
            if clas not in self.classes:
                print(f'{clas} not a valid class')
                return
        self._selected_classes = classes.copy()


    def connect_accl(self, dfp=None, num_classes=1, size='s'):
        print('Connecting to Accl...', flush=True, end='')
        if dfp is None:
            self._accl = memryx.AsyncAccl(dfp=self._model_path.with_suffix('.dfp'))
        else:
            self._accl = memryx.AsyncAccl(dfp=dfp)

        print('created...', flush=True, end='')
        self._accl.set_postprocessing_model(f'{self._model_path}_post.onnx')
        self._accl.connect_input(self._data_source)
        self._accl.connect_output(self._data_sink)
        print('connected...', flush=True, end='')

        with self._stop_lock:
            self._stopped = False

        print('Done.')

    def stop(self):
        print('Stopping')
        if self._stopped:
            print('Already Stopped!')
            return 

        with self._stop_lock:
            while self._outstanding_frames > 0:
                try:
                    self.get(timeout=0.1)
                except queue.Empty:
                    continue
                
            self.input_q.put(None)
            self._stopped = True

    def put(self, data,block=True, timeout=None):
        with self._stop_lock:
            if self._stopped:
                return
            else:
                self.input_q.put(data, block, timeout)
                self._outstanding_frames += 1

    def _data_source(self):
        data = self.input_q.get()
        if data is None:
            return None
        orig_frame = data.orig_frame
        self.do_seg = data.do_seg
        self.do_det = data.do_det 
        self.do_label = data.do_label
        self.rm_background = data.rm_background
        self.do_blur = data.do_blur

        padded_image, self._pads = letterbox(orig_frame, self._img_size)

        self.middle_q.put(orig_frame)

        ifmap = cv2.resize(padded_image, (self._img_size, self._img_size), interpolation=cv2.INTER_LINEAR)
        ifmap = ifmap.astype(np.float32) / 255.0 
        ifmap = np.transpose(ifmap, (2, 0, 1))
        ifmap = np.expand_dims(ifmap, axis=0)
        ifmap = ifmap.astype(np.float32)
        return ifmap

    def postprocess_dets(self):
        pass

    def _data_sink(self, *outputs):
        orig_frame = self.middle_q.get()
        preds = [torch.Tensor(outputs[1]),torch.Tensor(outputs[0])]
        dets = self._predictor.postprocess(preds, placeholder, [orig_frame])
        try:
            annotated_frame = self.plot_dets(orig_frame, dets[0],self._selected_classes,self.do_seg,self.do_det,self.do_label,self.rm_background,self.do_blur,self._darkness,self._blur_sigma)
            self.output_q.put(AnnotatedFrame(annotated_frame))
        except Exception as e:
            # print(f'Error in plotting {e}')
            self.output_q.put(AnnotatedFrame(orig_frame))
        
    def get(self, block=True, timeout=None):
        annotated_frame = self.output_q.get(block, timeout)
        self._outstanding_frames -= 1
        return annotated_frame

    def empty(self):
        return self.output_q.empty() and self.input_q.empty()  

    def shutdown(self):
        if self._accl:
            self._accl.shutdown()

    def update_classes(self, classes):
        times = []
        times.append(['start', time.time(), 0])
        self.stop() # Drains the pipeline.
        times.append(['drain', time.time(), 0])

        # update new model with classes
        self.yoloe.set_classes(classes, self.yoloe.get_text_pe(classes))

        self._predictor = setup_predictor(self.yoloe,
                                          imgsz=self._img_size,
                                          conf=self._conf, 
                                          iou=self._iou,
                                          classes=classes)
        self.classes = classes 
        self.selected_classes = classes

        times.append(['predictor setup', time.time(), 0])

        self.yoloe.export(imgsz=self._img_size, simplify=True, format='onnx')
        times.append(['onnx export', time.time(), 0])
        
        outs = 'output1, /model.22/cv5.0/cv5.0.2/Conv_output_0, /model.22/cv5.1/cv5.1.2/Conv_output_0,/model.22/cv5.2/cv5.2.2/Conv_output_0,/model.22/cv2.0/cv2.0.2/Conv_output_0,/model.22/cv3.0/cv3.0.1/act/Mul_output_0, /model.22/cv2.1/cv2.1.2/Conv_output_0,/model.22/cv3.1/cv3.1.1/act/Mul_output_0,/model.22/cv2.2/cv2.2.2/Conv_output_0,/model.22/cv3.2/cv3.2.1/act/Mul_output_0'
        nc = memryx.NeuralCompiler(f'yoloe-{self.version}{self.size}-seg.onnx',outputs=outs,no_sim_dfp=True)
        nc.load()

        # Move the post.onnx file to the assets directory
        post_onnx_path = Path(f'yoloe-{self.version}{self.size}-seg_post.onnx')
        target_path = self._assets_dir / post_onnx_path.name
        if post_onnx_path.exists():
            shutil.move(str(post_onnx_path), str(target_path))
        else:
            print(f"Warning: {post_onnx_path} does not exist.")

        self.connect_accl()
        times.append(['restart', time.time(), 0])

        # fill in the times
        for i in range(1,len(times)):
            times[i][2] = times[i][1] - times[i-1][1]

        for name, _, dt in times:
            print(f'{name:-<20}: {dt*1000:.1f} ms')

        print('-'*25)
        print(f'Total---------------: {(times[-1][1] - times[0][1])*1000:.1f} ms')

    @staticmethod
    def plot_dets(
        img,
        dets,
        filter=None,
        do_seg=True,
        do_det=True,
        do_label=True,
        remove_background=False,
        do_blur=False,
        alpha=0.25,
        blur_sigma=0
    ):
        """
        Display detections with three background modes:

        1) "normal" : original background
        2) "remove" : background is black, objects only
        3) "blur"   : background blurred, objects sharp

        Segmentation tint (do_seg), boxes (do_det), and labels (do_label)
        apply consistently in ALL modes.
        """ 
        classes = getattr(dets, "names", [])
        h, w = img.shape[:2]

        # --- 1) Build union mask and tint-only overlay ---
        overlay  = np.zeros_like(img)             # ONLY colored polygons (no image data)
        mask_all = np.zeros((h, w), dtype=np.uint8)

        # --- 2) Choose background exactly once ---
        if remove_background and do_blur:
            # hard rule: can't run both at once
            raise ValueError("remove_background and do_blur cannot both be True.")

        # blur only when sigma > 0; otherwise treat as normal background
        sigma = float(blur_sigma) if blur_sigma is not None else 0.0
        if remove_background:
            base = np.zeros_like(img)  # black background
        elif do_blur and sigma > 0.0:
            base = cv2.GaussianBlur(img, (0, 0), sigma)
        else:
            base = img  # normal background

        result = base.copy()

        # --- 3) Build masks + (optional) seg tint ---
        for cls_idx, xyxy, seg in zip(dets.boxes.cls, dets.boxes.xyxy, dets.masks):
            class_name = classes[int(cls_idx)] if classes else str(int(cls_idx))
            if filter is not None and class_name not in filter:
                continue
            if seg is None:
                continue

            # expecting polygon seg.xy (Nx2)
            poly = np.asarray(seg.xy, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_all, [poly], 255)  # union mask in-place

            if do_seg:
                color = COLOR_WHEEL[int(cls_idx) % len(COLOR_WHEEL)]
                cv2.fillPoly(overlay, [poly], color)  # paint only tint

        # --- 4) Paste objects onto chosen background ---
        if np.any(mask_all):
            if do_seg:
                # tint over the ORIGINAL image (not blurred)
                blended_region = cv2.addWeighted(img, 1 - float(alpha), overlay, float(alpha), 0)
                result[mask_all == 255] = blended_region[mask_all == 255]
            else:
                # no tint -> keep objects sharp from original image
                result[mask_all == 255] = img[mask_all == 255]
        # if no masks, background stays as-is (black/blur/normal)

        # --- 5) Draw boxes & labels last (stay crisp) ---
        if do_det or do_label:
            for cls_idx, xyxy, seg in zip(dets.boxes.cls, dets.boxes.xyxy, dets.masks):
                class_name = classes[int(cls_idx)] if classes else str(int(cls_idx))
                if filter is not None and class_name not in filter:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = COLOR_WHEEL[int(cls_idx) % len(COLOR_WHEEL)]

                if do_det:
                    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

                if do_label:
                    (tw, th), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                    text_x = max(min(cx - tw // 2, w - tw), 0)
                    text_y = max(min(cy + th // 2, h - baseline), th)
                    # Outline then color for readability
                    cv2.putText(result, class_name, (text_x, text_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(result, class_name, (text_x, text_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1, cv2.LINE_AA)

        return result


if __name__ == '__main__':
    yoloe = YoloE()

    image = cv2.imread('street.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    yoloe.put((np.array(image_rgb),False))
    annotated_frame = yoloe.get()
    
    plt.imshow(annotated_frame.image)
    plt.axis("off")  # Hide the axes
    plt.title("Image")
    plt.show()

    yoloe.update_classes(['person','bag', 'taxi'])

    yoloe.put((np.array(image_rgb), True))  # True for segmentation
    annotated_frame = yoloe.get()
    plt.imshow(annotated_frame.image)
    plt.axis("off")  # Hide the axes
    plt.title("Image")
    plt.show()

    yoloe.stop()

