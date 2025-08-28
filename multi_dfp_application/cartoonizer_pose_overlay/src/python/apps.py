import cv2 as cv
import numpy as np
from queue import Queue
import time

AVG_FPS_CALC_FRAME_COUNT = 50


class Cartoonizer:
    def __init__(
        self,
        frame_queue,
        accl,
        displayer,
        dfp_name,
        src_is_cam,
        stop_flag,
        scale=1.0,
    ):
        self.dfp_name = dfp_name
        self.displayer = displayer
        self.frame_queue = frame_queue
        self.scale = scale

        self.src_is_cam = src_is_cam
        self.stop_flag = stop_flag

        self.input_height = None
        self.input_width = None

        self.accl = accl
        self.accl.connect_input(self.get_frame)
        self.accl.connect_output(self.process_model_output)

        self.relay_frame_queue = Queue()

        # fps
        self.frame_count = 0
        self.start_ms = None
        self.fps_number = 0.0

    def get_frame(self):

        if self.stop_flag.is_set():
            return None

        while True:
            frame_id, frame = self.frame_queue.get()

            if self.input_height is None or self.input_width is None:
                self.input_height = int(frame.shape[0] * self.scale)
                self.input_width = int(frame.shape[1] * self.scale)

            self.relay_frame_queue.put((frame_id, frame))
            # print(f"[CARTOONIZER IN] Processing frame {frame_id}")
            return self.preprocess(frame)

    def preprocess(self, img):
        arr = cv.resize(img, (512, 512)).astype(np.float32)
        arr = arr / 127.5 - 1
        arr = np.expand_dims(arr, 0)
        return np.transpose(arr, (0, 3, 1, 2))

    def postprocess(self, frame, original_shape):
        frame = np.squeeze(frame, axis=0)
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame + 1) * 127.5
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return cv.resize(frame, original_shape)

    def process_model_output(self, *ofmaps):
        display_img = self.postprocess(ofmaps[0], (self.input_width, self.input_height))
        frame_id, frame = self.relay_frame_queue.get()

        # print(f"[CARTOONIZER OUT] process model output {frame_id}")

        self.displayer.update_buffer(frame_id, self.dfp_name, display_img)

        self.update_fps()

        return display_img

    def update_fps(self):

        self.frame_count += 1

        if self.frame_count == 1:
            self.start_ms = time.time() * 1000  # milliseconds
        elif self.frame_count % AVG_FPS_CALC_FRAME_COUNT == 0:
            now_ms = time.time() * 1000
            duration = now_ms - self.start_ms
            self.fps_number = AVG_FPS_CALC_FRAME_COUNT * 1000 / duration
            self.frame_count = 0
            # print(f"[CARTOONIZER] FPS: {self.fps_number:.2f}")


class PoseEstimation:
    def __init__(
        self,
        frame_queue,
        accl,
        displayer,
        dfp_name,
        src_is_cam,
        stop_flag,
        model_input_shape=(640, 640),
    ):
        self.dfp_name = dfp_name
        self.frame_queue = frame_queue
        self.displayer = displayer
        self.model_input_shape = model_input_shape
        self.src_is_cam = src_is_cam
        self.stop_flag = stop_flag

        self.input_height = None
        self.input_width = None
        self.relay_frame_queue = Queue()
        self.frame_id_queue = Queue()

        self.box_score = 0.25
        self.kpt_score = 0.5
        self.nms_thr = 0.2
        self.ratio = None

        self.COLOR_LIST = [
            [128, 255, 0],
            [255, 128, 50],
            [128, 0, 255],
            [255, 255, 0],
            [255, 102, 255],
            [255, 51, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 51, 51],
            [153, 255, 153],
            [51, 255, 51],
            [0, 255, 0],
            [255, 0, 51],
            [153, 0, 153],
            [51, 0, 51],
            [0, 0, 0],
            [0, 102, 255],
            [0, 51, 255],
            [0, 153, 255],
            [0, 153, 153],
        ]

        self.KEYPOINT_PAIRS = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 5),
            (0, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

        self.accl = accl
        self.accl.connect_input(self.get_frame)
        self.accl.connect_output(self.process_model_output)

        # fps
        self.frame_count = 0
        self.start_ms = None
        self.fps_number = 0.0

    def get_frame(self):

        if self.stop_flag.is_set():
            return None

        while True:
            frame_id, frame = self.frame_queue.get()

            if self.input_height is None or self.input_width is None:
                self.input_height = int(frame.shape[0])
                self.input_width = int(frame.shape[1])

            self.relay_frame_queue.put((frame_id, frame))
            out, self.ratio = self.preprocess_image(frame)

            # print(f"[POSE IN] Processing frame {frame_id}")

            return out

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        resized = cv.resize(image, (int(w * r), int(h * r)))
        padded = np.ones((*self.model_input_shape, 3), dtype=np.uint8) * 114
        padded[: int(h * r), : int(w * r)] = resized
        padded = padded / 255.0
        padded = np.transpose(padded.astype(np.float32), (2, 0, 1))
        return np.expand_dims(padded, axis=0), r

    def xywh2xyxy(self, box):
        box[..., 0] = box[..., 0] - box[..., 2] / 2
        box[..., 1] = box[..., 1] - box[..., 3] / 2
        box[..., 2] = box[..., 0] + box[..., 2]
        box[..., 3] = box[..., 1] + box[..., 3]
        return box

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        union_area = (
            (box[2] - box[0]) * (box[3] - box[1])
            + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            - inter_area
        )
        return inter_area / union_area

    def nms_process(self, boxes, scores, iou_thr):
        sorted_idx = np.argsort(scores)[::-1]
        keep_idx = []
        while sorted_idx.size > 0:
            idx = sorted_idx[0]
            keep_idx.append(idx)
            ious = self.compute_iou(boxes[idx], boxes[sorted_idx[1:]])
            sorted_idx = sorted_idx[np.where(ious < iou_thr)[0] + 1]
        return keep_idx

    def process_model_output(self, *ofmaps):

        frame_id, frame = self.relay_frame_queue.get()
        # print(f"[POSE OUT] process model output {frame_id}")
        display_img = frame.copy()

        predict = ofmaps[0].squeeze(0).T
        predict = predict[predict[:, 4] > self.box_score]
        scores = predict[:, 4]
        boxes = self.xywh2xyxy(predict[:, 0:4] / self.ratio)
        kpts = predict[:, 5:]

        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3 * j + 2] < self.kpt_score:
                    kpts[i, 3 * j : 3 * (j + 1)] = [-1, -1, -1]
                else:
                    kpts[i, 3 * j] /= self.ratio
                    kpts[i, 3 * j + 1] /= self.ratio

        idxes = self.nms_process(boxes, scores, self.nms_thr)

        for kpt in np.array(kpts)[idxes]:
            for pair in self.KEYPOINT_PAIRS:
                pt1 = kpt[3 * pair[0] : 3 * (pair[0] + 1)]
                pt2 = kpt[3 * pair[1] : 3 * (pair[1] + 1)]
                if pt1[2] > 0 and pt2[2] > 0:
                    cv.line(
                        display_img,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (255, 255, 255),
                        3,
                    )
            for idx in range(len(kpt) // 3):
                x, y, score = kpt[3 * idx : 3 * (idx + 1)]
                if score > 0:
                    cv.circle(
                        display_img,
                        (int(x), int(y)),
                        5,
                        self.COLOR_LIST[idx % len(self.COLOR_LIST)],
                        -1,
                    )

        self.displayer.update_buffer(frame_id, self.dfp_name, display_img)

        self.update_fps()

        return display_img

    def update_fps(self):

        self.frame_count += 1

        if self.frame_count == 1:
            self.start_ms = time.time() * 1000  # milliseconds
        elif self.frame_count % AVG_FPS_CALC_FRAME_COUNT == 0:
            now_ms = time.time() * 1000
            duration = now_ms - self.start_ms
            self.fps_number = AVG_FPS_CALC_FRAME_COUNT * 1000 / duration
            self.frame_count = 0
            # print(f"[POSE] FPS: {self.fps_number:.2f}")