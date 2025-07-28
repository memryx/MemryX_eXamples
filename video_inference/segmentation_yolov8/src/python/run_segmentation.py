import sys
import os
import argparse
os.environ["OMP_NUM_THREADS"] = "1"  # Set number of threads to 1 for performance control
from pathlib import Path
from queue import Queue, Full
import cv2 as cv
import numpy as np
from memryx import AsyncAccl

from ultralytics.utils import ops
import torchvision.ops as ops
from typing import List, Tuple, Dict
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Colors

import time
class App:
    def __init__(self, cam, use_tflite, display=True, mirror=False, src_is_cam=True, **kwargs):
        # Initialize camera, display settings, input resolution, and other variables
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.model_input_shape = (640, 640)  # Model input size
        self.capture_queue = Queue(maxsize=5)  # Queue to store captured frames
        self.mirror = mirror
        self.box_score = 0.25
        self.ratio = None
        self.pad_w = None 
        self.pad_h = None
        self.iou_threshold = 0.45
        self.conf_threshold = 0.25
        self.display = display
        self.src_is_cam = src_is_cam
        self.use_tflite = use_tflite

        self.color_palette = Colors()  # Set color palette for drawing
        # Load COCO class names from a yaml file
        self.classes = YAML.load(check_yaml("coco8.yaml") )["names"]

    def _free(self, cap):
        # Release the camera and allow a clean exit
        cap.release()
        time.sleep(0.5)

    def generate_frame(self):
        # Capture a frame from the camera, preprocess and add to queue
        while True:
            ok, frame = self.cam.read()
            if not ok:
                print('EOF')
                return None
            if self.src_is_cam and self.capture_queue.full():
                # drop frame and move on
                continue
            else:
                if self.mirror:
                    frame = cv.flip(frame, 1)  # Flip frame if mirror is True
                self.capture_queue.put(frame)  # Put the captured frame in the queue
                out, self.ratio, (self.pad_w, self.pad_h) = self.preprocess_image(frame)
                return out

    def preprocess_image(self, image):
        # Preprocess the image (resize, pad, normalize) for the model input
        input_shape = self.model_input_shape
        h, w = image.shape[:2]
        r = min(input_shape[0] / h, input_shape[1] / w)
        new_unpad = (int(w * r), int(h * r))
        pad_w, pad_h = (input_shape[1] - new_unpad[0]) / 2, (input_shape[0] - new_unpad[1]) / 2

        if (w, h) != new_unpad:
            image = cv.resize(image, new_unpad, interpolation=cv.INTER_LINEAR)
        
        # Pad the image and normalize
        padded_img = np.ones((input_shape[0], input_shape[1], 3), dtype=np.uint8) * 114
        padded_img[int(pad_h):int(pad_h) + new_unpad[1], int(pad_w):int(pad_w) + new_unpad[0], :] = image
        padded_img = padded_img / 255.0  # Normalize to [0, 1]

        if not self.use_tflite:
            padded_img = np.transpose(padded_img, (2,0,1))   # Transpose to bring it to the expected ONNX model shape 
            padded_img = np.expand_dims(padded_img, axis=0)  # Add batch dimension
        else:
            padded_img = np.expand_dims(padded_img, axis=0)  # Add batch dimension

        img_process = padded_img.astype(np.float32)

        return img_process, (r, r), (pad_w, pad_h)

    def process_model_output(self, *ofmaps):
        # Process model outputs, extract boxes, segments, and masks
        out = ofmaps[0]
        protos = ofmaps[1]

        # Retrieve the image from the queue
        img = self.capture_queue.get()
        self.capture_queue.task_done()

        # Post-process the outputs to obtain boxes, segments, and masks
        boxes, segments, masks = self.postprocess(protos, out, img, self.ratio, self.pad_w, self.pad_h, self.conf_threshold, self.iou_threshold)

        # Draw bboxes and polygons
        if len(boxes) > 0:
            img = self.draw_and_visualize(img, boxes, segments)
        
        self.show(img)  # Display the output image
        return img

    def postprocess(self, output0, output1, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        # Post-process the model output to extract bounding boxes, segments, and masks

        x, protos = output0, output1   # Two outputs: predictions and protos
            
        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Filter predictions by confidence threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Merge bounding box, score, class, and mask data
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # Perform Non-Max Suppression (NMS)
        x = x[cv.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        if len(x) > 0:
            # Convert boxes from center format (cxcywh) to corner format (xyxy)
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescale boxes to original image size
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Clamp box boundaries to image dimensions
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process mask data
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # Convert masks to segments
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):
        # Convert masks to contour segments
        segments = []
        for x in masks.astype("uint8"):
            c = cv.findContours(x, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        # Crop the masks to the bounding boxes
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        
        # align proto shape
        if self.use_tflite: # use tflite
            protos = np.einsum("hwc->chw", protos)
        
        # Process the mask outputs from the model
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # Scale the mask to original input image size
        masks = np.einsum("HWN -> NHW", masks)  # Reshape masks
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        # Scale the mask from model output size to the original image size
        im1_shape = masks.shape[:2]
        
        if ratio_pad is None:
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain = old / new
            pad_w = (im1_shape[1] - im0_shape[1] * gain) / 2
            pad_h = (im1_shape[0] - im0_shape[0] * gain) / 2
        else:
            pad_w, pad_h = ratio_pad[1]
        
        top = int(round(pad_h - 0.1))
        left = int(round(pad_w - 0.1))
        bottom = int(round(im1_shape[0] - pad_h + 0.1))
        right = int(round(im1_shape[1] - pad_w + 0.1))

        # Slice and resize the mask
        masks = masks[top:bottom, left:right]
        masks = cv.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv.INTER_LINEAR)

        # Ensure mask has 3 dimensions
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=-1)

        return masks

    def draw_and_visualize(self, im, bboxes, segments):
        # Draw bounding boxes and polygons on the image
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # Draw the contour and fill mask
            cv.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # White border line
            cv.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            # Draw bounding box
            cv.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv.LINE_AA,
            )
            # Add label text
            cv.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv.LINE_AA,
            )

        # Blend the mask and original image
        im = cv.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        return im

    def show(self, img):
        # Display the output image
        if not self.display:
            return
        
        cv.imshow('Output', img)
        if cv.waitKey(1) == ord('q'):
            self._free(self.cam)  # Release resources and exit
            exit(1)
            
def run_mxa(dfp, post_model, app):
    # Run the model inference using the Memryx AsyncAccl
    accl = AsyncAccl(dfp)
    accl.set_postprocessing_model(post_model, model_idx=0)  # Set the post-processing model
    accl.connect_input(app.generate_frame)  # Connect the input generator (frames)
    accl.connect_output(app.process_model_output)  # Connect the output processing
    accl.wait()  # Wait for async processing to complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Inference")
    
    # Add arguments for the DFP file and post-processing model
    parser.add_argument('-d', '--dfp', type=str, default='models/onnx/YOLO_v8_nano_seg_640_640_3_onnx.dfp', help='Path to the compiled DFP file (default: models/YOLO_v8_nano_seg_640_640_3_onnx.dfp)')
    parser.add_argument('-p', '--post_model', type=str, default='models/onnx/YOLO_v8_nano_seg_640_640_3_onnx_post.onnx', help='Path to the post-processing ONNX file (default: models/YOLO_v8_nano_seg_640_640_3_onnx_post.onnx)')
    
    args = parser.parse_args()

    cam = cv.VideoCapture(0)  # Open video capture (webcam)
    parent_path = Path(__file__).resolve().parent

    use_tflite = args.post_model.endswith('.tflite')  # Check if the post-processing model is a TFLite model
    app = App(cam, use_tflite, mirror=False, src_is_cam=True)  # Initialize the application
    dfp = Path(args.dfp)  # Get DFP path from argument
    post_model = str(Path(args.post_model))  # Get post-processing model path from argument

    run_mxa(dfp, post_model, app)  # Run the application
    cv.destroyAllWindows()  # Clean up windows
