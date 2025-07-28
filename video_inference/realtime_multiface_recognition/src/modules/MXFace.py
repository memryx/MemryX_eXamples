import queue
import logging
from pathlib import Path
import cv2
import memryx as mx
import numpy as np
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class Face:
    """container for passing a face around"""
    track_id: int
    face: np.ndarray


@dataclass
class DetectedFace():
    # Bounding box coords (left,top,width,height)
    bbox: tuple[int, int, int, int] = field(default_factory=lambda: (0,0,0,0))

    # [(kx, ky), ...]
    keypoints: list[tuple[int, int]] = field(default_factory=lambda: [])

    # np.array([height, width, 3])
    image: np.ndarray = field(default_factory=lambda: np.ndarray([0,0,3]))

    # Embedding
    embedding: np.ndarray = field(default_factory=lambda: np.zeros([128]))


@dataclass
class AnnotatedFrame():
    image: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=lambda: [])
    keypoints: list[list[tuple[int,int]]] = field(default_factory=lambda: [])
    scores: list = field(default_factory=lambda: [])


    @property
    def num_detected_faces(self):
        return len(self.scores)


class MXFace():
    detector_imgsz = 640
    recognizer_imgsz = 160 

    def __init__(self, models_dir: Path):
        self._stopped = False
        self.do_eye_alignment = True

        self.detect_input_q  = queue.Queue(maxsize=1)
        self.detect_output_q  = queue.Queue(maxsize=1)
        self.detect_bypass_q = queue.Queue(maxsize=4)

        self.recognize_input_q  = queue.Queue(maxsize=1)
        self.recognize_output_q  = queue.Queue(maxsize=1)
        self.recognize_bypass_q = queue.Queue(maxsize=4)

        self.accl = mx.AsyncAccl(str(Path(models_dir) / 'yolov8n_facenet.dfp'), local_mode=True)
        self.accl.set_postprocessing_model(str(Path(models_dir) / 'yolov8n-face_post.onnx'), model_idx=1)

        self.accl.connect_input(self._detector_source, model_idx=1)
        self.accl.connect_output(self._detector_sink, model_idx=1)
        self.accl.connect_input(self._recognizer_source, model_idx=0)
        self.accl.connect_output(self._recognizer_sink, model_idx=0)

        self._outstanding_detection_frames = 0
        self._outstanding_recognition_frames = 0

    ### Public Functions ######################################################

    def stop(self):
        logger.info('stop')
        print('Shutting down MXFace')

        while self._outstanding_detection_frames > 0:
            try:
                self.detect_get(timeout=0.1)
            except queue.Empty:
                print('outstanding detection timeout')
                continue
        self.detect_input_q.put(None)
        print('detection drained')

        while self._outstanding_recognition_frames > 0:
            try:
                self.recognize_get(timeout=0.1)
            except queue.Empty:
                print('outstanding recognition timeout')
                continue
        print('recognized drained')
        self.recognize_input_q.put((None, None))

        self._stopped = True
        #self.accl.shutdown()

    def detect_put(self, image, block=True, timeout=None):
        annotated_frame = AnnotatedFrame(np.array(image))
        self.detect_input_q.put(annotated_frame, block, timeout)
        self._outstanding_detection_frames += 1

    def detect_get(self, block=True, timeout=None):
        annotated_frame = self.detect_output_q.get(block, timeout)
        self._outstanding_detection_frames -= 1
        return annotated_frame

    def recognize_put(self, obj, block=True, timeout=None):
        (id, frame, box, eyes) = obj
        face = self._extract_face(frame, box, eyes)
        self.recognize_input_q.put((id, face), block, timeout)
        self._outstanding_recognition_frames += 1

    def recognize_get(self, block=True, timeout=None):
        labeled_embedding = self.recognize_output_q.get(block, timeout)
        self._outstanding_recognition_frames -= 1
        return labeled_embedding

    def _extract_face(self, 
                      image: np.ndarray, 
                      xyxy: tuple[int, int, int, int], 
                      eyes: tuple[tuple[int,int]]) -> np.ndarray:
        x1, y1, x2, y2 = xyxy
        orig_h, orig_w, _ = image.shape
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), orig_w)
        y2 = min(int(y2), orig_h)
        face = image[y1:y2, x1:x2]

        if self.do_eye_alignment:
            face, bbox = self._align_eyes(face, xyxy, eyes)

        return face

    def _align_eyes(self, image: np.ndarray, bbox, eyes):
        right_eye = eyes[0]
        left_eye = eyes[1]
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        rotation_angle = angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.transform(corners, M).reshape(-1, 2)
        x_min = int(np.min(transformed[:, 0]))
        y_min = int(np.min(transformed[:, 1]))
        x_max = int(np.max(transformed[:, 0]))
        y_max = int(np.max(transformed[:, 1]))
        new_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return rotated_image, new_bbox


    ### Async Functions #######################################################
    def _detector_source(self):
        annotated_frame = self.detect_input_q.get()
        
        if annotated_frame is None:
            return None

        self.detect_bypass_q.put(annotated_frame)

        ifmap = self._letterbox_image(
            annotated_frame.image, 
            (self.detector_imgsz, self.detector_imgsz)
        ) 
        ifmap = ifmap / 255.0
        ifmap = np.transpose(ifmap, (2, 0, 1))
        ifmap = np.expand_dims(ifmap, 0)
        return ifmap.astype(np.float32)

    def _detector_sink(self, *outputs):
        annotated_frame = self.detect_bypass_q.get()
        image = annotated_frame.image
        detections = self._postprocess_detector(image, outputs[0])

        annotated_frame.boxes = detections['boxes']
        annotated_frame.keypoints = detections['keypoints']
        annotated_frame.scores = detections['scores'] 
        self.detect_output_q.put(annotated_frame)

    def _recognizer_source(self):
        track_id, detected_face = self.recognize_input_q.get()

        if detected_face is None:
            return None

        self.recognize_bypass_q.put(track_id)

        face = self._letterbox_image(
            detected_face, 
            (self.recognizer_imgsz, self.recognizer_imgsz)
        )
        face = face / 255.0
        face = np.expand_dims(face, 0)
        return face.astype(np.float32)

    def _recognizer_sink(self, *outputs):
        track_id = self.recognize_bypass_q.get()
        embedding = np.squeeze(outputs[0])
        self.recognize_output_q.put((track_id, embedding))

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


class MockMXFace(MXFace):

    def __init__(self, models_dir: Path):
        self.detect_q  = queue.Queue(maxsize=1)
        self.recognize_q  = queue.Queue(maxsize=1)

    def stop(self):
        pass

    def detect_put(self, image, block=True, timeout=None):
        annotated_frame = AnnotatedFrame(np.array(image))
        self.detect_q.put(annotated_frame, block, timeout)

    def detect_get(self, block=True, timeout=None):
        annotated_frame = self.detect_q.get(block, timeout)
        return annotated_frame

    def recognize_put(self, obj, block=True, timeout=None):
        """obj: (track_id, frame, xyxy, eyes)"""
        self.recognize_q.put(obj, block, timeout)

    def recognize_get(self, block=True, timeout=None):
        (id, _) = self.recognize_q.get(block, timeout)
        return (id, np.zeros([128]))

