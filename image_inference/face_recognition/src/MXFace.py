import queue
import logging
from pathlib import Path
import cv2
import memryx as mx
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace():
    # Bounding box coords [(left,top,width,height), ...]
    bbox: list[tuple[int, int, int, int]] = field(default_factory=lambda: [])

    # [(kx, ky), ...]
    keypoints: list[tuple[int, int]] = field(default_factory=lambda: [])

    # np.array([height, width, 3])
    image: np.ndarray = field(default_factory=lambda: np.ndarray([0,0,3]))

    # Embedding
    embedding: np.ndarray = field(default_factory=lambda: np.zeros([128]))

@dataclass
class AnnotatedFrame():
    image: np.ndarray
    num_detections: int = 0
    detected_faces: list[DetectedFace] = field(default_factory=lambda: [])
    recognize: bool = True
    _static: bool = False

class MXFace():
    cosine_threshold = 0.48
    detector_imgsz = 640
    recognizer_imgsz = 160 

    def __init__(self, models_dir=None):
        self._stopped = False
        self._outstanding_frames = 0

        self.input_q  = queue.Queue(maxsize=1)
        self.stage0_q = queue.Queue(maxsize=2)
        self.stage1_q = queue.Queue(maxsize=4)
        self.stage2_q = queue.Queue(maxsize=4)
        self.output_q = queue.Queue(maxsize=1)
        self.static_output_q = queue.Queue(maxsize=1)

        self.accl = mx.AsyncAccl(str(Path(models_dir) / 'yolov8n_facenet.dfp'))
        self.accl.set_postprocessing_model(str(Path(models_dir) / 'yolov8n-face_post.onnx'), model_idx=1)

        self.accl.connect_input(self._detector_source, model_idx=1)
        self.accl.connect_output(self._detector_sink, model_idx=1)
        self.accl.connect_input(self._recognizer_source, model_idx=0)
        self.accl.connect_output(self._recognizer_sink, model_idx=0)

    def __del__(self):
        if not self._stopped:
            self.stop()

    ### Public Functions ######################################################
    @staticmethod
    def cosine_similarity(vector1, vector2):
        # Ensure the vectors are numpy arrays
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        
        # Compute the dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        # Handle the case where the magnitude is zero to avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Compute cosine similarity
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        
        return cosine_sim

    def infer(self, image):
        annotated_frame = AnnotatedFrame(np.array(image), _static=True)
        self.input_q.put(annotated_frame)
        return self.static_output_q.get()

    def stop(self):
        logger.info('stop')
        while self._outstanding_frames > 0:
            try:
                self.get(timeout=0.1)
            except queue.Empty:
                continue
            
        self.input_q.put(None)
        self.stage1_q.put(None)
        self._stopped = True

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

        ifmap = self._letterbox_image(
            annotated_frame.image, 
            (self.detector_imgsz, self.detector_imgsz)
        ) 
        ifmap = ifmap / 255.0
        return ifmap.astype(np.float32)

    def _detector_sink(self, *outputs):
        annotated_frame = self.stage0_q.get()
        image = annotated_frame.image
        detections = self._postprocess_detector(image, outputs[0])

        # Count found faces
        annotated_frame.num_detections = len(detections['boxes'])
        if  annotated_frame.num_detections == 0:
            self.output_q.put(annotated_frame)
            return 

        # Extract Faces
        boxes = detections['boxes'] 
        keypoints = detections['keypoints'] 
        detected_faces = []
        for i, (bbox, kpts) in enumerate(zip(boxes, keypoints)):
            detected_faces.append(DetectedFace(bbox, kpts))

        if annotated_frame.recognize:
            for detected_face in detected_faces:
                self.stage1_q.put((annotated_frame, detected_face))
        else:
            annotated_frame.detected_faces = detected_faces
            if annotated_frame._static:
                self.static_output_q.put(annotated_frame)
            else:
                self.output_q.put(annotated_frame)

    def _recognizer_source(self):
        data = self.stage1_q.get()

        if data is None:
            return None

        # Extract
        annotated_frame, detected_face = data
        detected_face.image = self._extract_face(annotated_frame.image, detected_face)

        # Put to final stage
        self.stage2_q.put(data)

        face = self._letterbox_image(
            detected_face.image, 
            (self.recognizer_imgsz, self.recognizer_imgsz)
        )
        face = face / 255.0
        return face.astype(np.float32)

    def _recognizer_sink(self, *outputs):
        annotated_frame, detected_face = self.stage2_q.get()
        detected_face.embedding = np.squeeze(outputs[0])
        annotated_frame.detected_faces.append(detected_face)

        if len(annotated_frame.detected_faces) == annotated_frame.num_detections:
            if annotated_frame._static:
                self.static_output_q.put(annotated_frame)
            else:
                self.output_q.put(annotated_frame)

    ### Pre / Post Processing steps ###########################################
    def _letterbox_image(self, image, target_size):
        original_size = image.shape[:2]
        ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
        
        # Calculate new size preserving the aspect ratio
        new_size = (int(original_size[1] * ratio), int(original_size[0] * ratio))
        
        # Resize the image
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Create a blank canvas with the target size
        canvas = np.full((target_size[1], target_size[0], 3), (128, 128, 128), dtype=np.uint8)  # Gray letterbox
        
        # Calculate padding for centering the resized image on the canvas
        top = (target_size[1] - new_size[1]) // 2
        left = (target_size[0] - new_size[0]) // 2
        canvas[top:top + new_size[1], left:left + new_size[0]] = resized_image
        
        return canvas

    def _adjust_coordinates(self, image, bbox, kpts):
        # Unpack the bounding box
        x, y, w, h = bbox
        
        # Get the original image dimensions
        orig_h, orig_w, _ = image.shape
    
        # The letterboxed image is 640x640, so calculate the aspect ratios
        aspect_ratio_original = orig_w / orig_h
        if aspect_ratio_original > 1:
            # Width is greater than height (landscape)
            new_w = self.detector_imgsz
            new_h = int(self.detector_imgsz / aspect_ratio_original)
            pad_y = (self.detector_imgsz - new_h) // 2  # Padding added to the top and bottom
            pad_x = 0
        else:
            # Height is greater than width (portrait)
            new_h = self.detector_imgsz
            new_w = int(self.detector_imgsz * aspect_ratio_original)
            pad_x = (self.detector_imgsz - new_w) // 2  # Padding added to the left and right
            pad_y = 0
    
        # Adjust the bounding box coordinates to remove the padding
        x_adj = (x - pad_x) / new_w * orig_w
        y_adj = (y - pad_y) / new_h * orig_h
        w_adj = w / new_w * orig_w
        h_adj = h / new_h * orig_h
        bbox = (int(x_adj), int(y_adj), int(w_adj), int(h_adj))

        # Adjust the keypoints coordinates to remove the padding
        new_kpts = []
        for x, y in kpts:
            x_adj = (x - pad_x) / new_w * orig_w
            y_adj = (y - pad_y) / new_h * orig_h
            new_kpts.append((int(x_adj), int(y_adj)))

        return bbox, new_kpts

    def _extract_face(self, image: np.ndarray, detected_face) -> np.ndarray:
        """
        """
        # Unpack the bounding box
        x, y, w, h = detected_face.bbox
    
        # Get the original image dimensions
        orig_h, orig_w, _ = image.shape

        # Compute the new top-left and bottom-right corners in the original image
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + w), orig_w)
        y2 = min(int(y + h), orig_h)
    
        # Extract the face from the original image using the adjusted bounding box
        face = image[y1:y2, x1:x2]

        return face

    def _postprocess_detector(self, image, output, conf_threshold=0.7, nms_threshold=0.7):
        """
        Processes the raw YOLOv8-face model output into a dictionary of bounding boxes and keypoints with NMS.
    
        Args:
        - image (np.array): original image (needed for original shape).
        - output (np.array): Raw output from YOLOv8-face model (1, 20, 8400).
        - conf_threshold (float): Confidence threshold for filtering detections.
        - nms_threshold (float): Intersection-over-Union (IoU) threshold for NMS.
    
        Returns:
        - dict: A dictionary containing bounding boxes and keypoints after applying NMS.
          Format:
          {
              "boxes": [(x1, y1, w, h)],  # List of bounding boxes as top-left and bottom-right corners
              "keypoints": [[(kp1_x, kp1_y), (kp2_x, kp2_y), ..., (kp5_x, kp5_y)]],  # List of 5 keypoints per box
              "scores": [confidence_scores]  # Confidence scores for each detection
          }
        """
        # Squeeze the output to remove extra dimensions (e.g., (1, 20, 8400) -> (20, 8400))
        output = output.squeeze()
    
        final_boxes = []
        final_keypoints = []
        final_scores = []

        conf_mask = output[4] > conf_threshold
        output = output[:, conf_mask]
        if output.shape[-1] == 0:
            return {"boxes": [], "keypoints": [], "scores": []}

        boxes = output[:4,:]
        scores = output[4,:]
        keypoints = output[5:, :]
    
        # Apply Non-Maximum Suppression (NMS)
        indices = self._nms(boxes, scores, nms_threshold)

        boxes = boxes[:, indices]
        scores = scores[indices]
        keypoints = keypoints[:, indices]

        # Process the output and extract bounding boxes, keypoints, and confidence scores
        for bbox, confidence, keypoints in zip(boxes.T, scores.T, keypoints.T):
            # Extract bounding box center, width, height, and confidence
            x_center, y_center, width, height = bbox
    
            # Calculate top-left
            x1 = x_center - width / 2
            y1 = y_center - height / 2

            # bbox as (t,l,w,h)
            bbox = (x1,y1,width,height)

            # Adjust keypoints and box to original image coordinates 
            kpts = keypoints.reshape(5, 3)[:, :2].tolist()
            adj_bbox, adj_kpts = self._adjust_coordinates(image, bbox, kpts)
    
            # Append bounding box, keypoints, and confidence
            final_boxes.append(adj_bbox)
            final_keypoints.append(adj_kpts)
            final_scores.append(confidence)
    
        return {"boxes": final_boxes, "keypoints": final_keypoints, "scores": final_scores}
    
    def _nms(self, boxes, scores, iou_threshold):
        """
        Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes based on IoU.
    
        Args:
        - boxes (np.array): Array of bounding boxes with shape (N, 4) where each box is [x1, y1, x2, y2].
        - scores (np.array): Array of confidence scores with shape (N,).
        - iou_threshold (float): IoU threshold for NMS.
    
        Returns:
        - np.array: Indices of the boxes to keep after applying NMS.
        """
        x1 = boxes[0, :] - boxes[2, :] / 2
        y1 = boxes[1, :] - boxes[3, :] / 2
        x2 = x1 + boxes[2, :] / 2
        y2 = y1 + boxes[3, :] / 2
    
        # Compute area of the bounding boxes
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]  # Sort by confidence scores in descending order
    
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
    
            # Compute IoU of the remaining boxes with the box with the highest score
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
    
            iou = inter / union
            indices_to_keep = np.where(iou <= iou_threshold)[0]
    
            order = order[indices_to_keep + 1]  # Update the order by excluding the boxes with high IoU
    
        return np.array(keep)

