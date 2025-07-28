from pathlib import Path
from abc import abstractmethod
import numpy as np
import cv2

import onnx, onnxruntime
onnxruntime.set_default_logger_severity(3)

COCO_CLASSES = ( "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",)

##  YoloX Base  ###############################################################
class YoloX:
    """
    Parent YoloX helper class for wrapping common pre/post processing steps.
    Each instantiation (nano,tiny,small,medium) will inherit from this class
    and properly set their own model parameters.
    """

    ##  Initialization  #######################################################
    def __init__(self, img_size=None):
        self.models_dir = Path(__file__).resolve().parent / 'models'
        self.set_model_params()
        if img_size:
            self.scale = min(self.input_size[0] / float(img_size[1]),
                    self.input_size[1] / float(img_size[0]))
        else:
            self.scale = 1


    @abstractmethod
    def set_model_params(self):
        self.model = None
        self.dfp_path = None
        self.input_size = None
        self.output_size = None
        self.output_order_map = None

    ##  Pre-Processing  #######################################################
    def preprocess(self, img):
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3),
                dtype=np.uint8) * 114

        self.scale = min(self.input_size[0] / float(img.shape[0]),
                self.input_size[1] / float(img.shape[1]))
        sx,sy = int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)

        resized_img = cv2.resize(img, (sx,sy), interpolation=cv2.INTER_LINEAR)
        padded_img[:sy, :sx] = resized_img.astype(np.uint8)
        

        # Step 4: Slice the padded image into 4 quadrants and concatenate them into 12 channels
        x0 = padded_img[0::2, 0::2, :]  # Top-left
        x1 = padded_img[1::2, 0::2, :]  # Bottom-left
        x2 = padded_img[0::2, 1::2, :]  # Top-right
        x3 = padded_img[1::2, 1::2, :]  # Bottom-right

        # Step 5: Concatenate along the channel dimension (axis 2)
        concatenated_img = np.concatenate([x0, x1, x2, x3], axis=2)

        # Step 6: Reshape image to what is expected by the original onnx model - i.e channel first, and a single batch
        concatenated_img = np.transpose(concatenated_img, (2,0,1))
        concatenated_img = np.expand_dims(concatenated_img, axis=0)

        # Step 7: Return the processed image as a contiguous array of type float32
        return np.ascontiguousarray(concatenated_img).astype(np.float32)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:

        return 1 / (1 + np.exp(-x))

    def onnx_concat(self, inputs: list, axis: int) -> np.ndarray:

        # Ensure all inputs are numpy arrays
        if not all(isinstance(x, np.ndarray) for x in inputs):
            raise TypeError("All inputs must be numpy arrays.")
        
        # Ensure shapes match on non-concat axes
        ref_shape = list(inputs[0].shape)
        for i, tensor in enumerate(inputs[1:], start=1):
            for ax in range(len(ref_shape)):
                if ax == axis:
                    continue
                if tensor.shape[ax] != ref_shape[ax]:
                    raise ValueError(f"Shape mismatch at axis {ax} between input[0] and input[{i}]")

        return np.concatenate(inputs, axis=axis)

    def onnx_reshape(self, data: np.ndarray, shape: np.ndarray) -> np.ndarray:

        # Ensure shape is a 1D array of integers
        target_shape = shape.astype(int).tolist()

        # Use NumPy reshape with dynamic handling of -1
        reshaped = np.reshape(data, target_shape)

        return reshaped

    ##  Post-Processing  ######################################################
    def postprocess(self, output):

        # output is of shape (1, C, H, W)

        output_785 = output[0]     # 785
        output_794 = output[1]     # 794   
        output_795 = output[2]     # 795
        output_811 = output[3]     # 811
        output_820 = output[4]     # 820
        output_821 = output[5]     # 821
        output_837 = output[6]     # 837
        output_846 = output[7]     # 846
        output_847 = output[8]     # 847

        output_795 = self.sigmoid(output_795)
        output_785 = self.sigmoid(output_785)
        output_821 = self.sigmoid(output_821)
        output_811 = self.sigmoid(output_811)
        output_847 = self.sigmoid(output_847)
        output_837 = self.sigmoid(output_837)

        concat_1 = self.onnx_concat([output_794, output_795, output_785], axis=1)
        concat_2 = self.onnx_concat([output_820, output_821, output_811], axis=1)
        concat_3 = self.onnx_concat([output_846, output_847, output_837], axis=1)

        shape = np.array([1, 85, -1], dtype=np.int64)   

        reshape_1 = self.onnx_reshape(concat_1, shape)
        reshape_2 = self.onnx_reshape(concat_2, shape)
        reshape_3 = self.onnx_reshape(concat_3, shape)

        concat_out = self.onnx_concat([reshape_1, reshape_2, reshape_3], axis=2)

        output = concat_out.transpose(0,2,1)  #1, 840, 85

        self.num_classes = output.shape[2] - 5
        
        post_output = output

        # Perform more post + NMS
        grids, expanded_strides = [], []
        strides = [8, 16, 32]

        hsizes = [self.input_size[0] // stride for stride in strides]
        wsizes = [self.input_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)

        post_output[..., :2] = (post_output[..., :2] + grids) * expanded_strides
        post_output[..., 2:4] = np.exp(post_output[..., 2:4]) * expanded_strides
        dets = self.convert(post_output[0])
        return dets

    ## Post-processing helpers ################################################
    def convert(self, output):
        boxes = output[:, 0:4]
        scores = output[:, 4:5] * output[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= self.scale

        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.5)
        if dets is None:
            return []

        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

        dets_list = []
        for i, arr in enumerate(dets):
            x1,y1,x2,y2 = arr[0:4].astype(int)
            det = {}
            det['bbox'] = (x1,y1,x2,y2)
            det['class'] = COCO_CLASSES[int(arr[5])]
            det['class_idx'] = int(arr[5])
            det['score'] = arr[4]
            dets_list.append(det)

        return dets_list

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

##  YoloXM class ####################################

class YoloXM(YoloX):
    def set_model_params(self):
        self.name = 'yolox-m'
        self.input_size = (640,640,3)
        self.output_size = [(80,80,85), (40,40,85), (20,20,85)]
        self.output_order_map = [0,1,2]

    def __init__(self, img_size=None, dfp_path=None):
        super().__init__(img_size)  # Call the base class constructor
        if dfp_path:
            self.dfp_path = dfp_path

def main():
    """
    Quick test on a single image.
    """

    # Load
    img = cv2.imread('samples/image.jpg')
    if img is None:
        raise RuntimError("Failed to load image")

    # Preprocess
    model = YoloXM()

    fmap = model.preprocess(img)
    fmap = model.run(fmap)
    dets = model.postprocess(fmap)

    for d in dets:
        print(d['class'], d['bbox'])

    for det in dets:
        x1,y1,x2,y2 = det['bbox']
        img = cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0))

    cv2.imshow('window', img.astype(np.uint8))
    cv2.waitKey(-1)

if __name__=="__main__":
    main()
