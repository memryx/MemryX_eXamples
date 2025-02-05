import copy
import math
import argparse
import sys
import cv2 as cv
import numpy as np
from pathlib import Path
from queue import Queue
import os


sys.path.append(str(Path(__file__).resolve().parent))


try:
    import memryx
except ImportError:
    mix_home = os.getenv("MIX_HOME")
    if not mix_home:
        print("Install MemryX SDK or clone MIX and source setup_env.sh")
        exit(1)
    sys.path.append(mix_home)

from memryx import AsyncAccl


class App:
    def __init__(self, cam, input_size, fps,score_th=0.5, nms_th=0.5, **kwargs):
        self.cam = cam
        self.input_size =  input_size
        self.npy_file = 'assets/0.npy'
        self.score_th=score_th
        self.nms_th=nms_th
        self.model_dfp = 'models/Vehicle_Detection_0200_256_256_3_tflite.dfp'
        self.pre_model = 'models/Vehicle_Detection_0200_256_256_3_tflite_pre.tflite'
        self.post_model = 'models/Vehicle_Detection_0200_256_256_3_tflite_post.tflite'
        self.capture_queue = Queue(maxsize=10)
        self.fps = fps
        self.fourcc = cv.VideoWriter_fourcc(*'mp4v')  
        self.frame_height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.out_video = cv.VideoWriter('output.mp4', self.fourcc, self.fps, (self.frame_width,self.frame_height))

    def run_mxa(self):
        accl = AsyncAccl(self.model_dfp)
        accl.set_preprocessing_model(self.pre_model, model_idx=0)
        accl.set_postprocessing_model(self.post_model, model_idx=0)
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess_and_display)
        accl.wait()
        cv.destroyAllWindows()
        accl.stop()

    def capture_and_preprocess(self):
        ret, frame = self.cam.read()

        if not ret:
            return None
        self.capture_queue.put(copy.deepcopy(frame),block=True)
        input_image = cv.resize(frame, dsize=(self.input_size, self.input_size))
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        return input_image


    def postprocess_and_display(self,*mxa_output):
        nms_bbox_list, nms_score_list = self.postprocess(*mxa_output)
        frame = self.capture_queue.get()
        self.draw(frame, nms_bbox_list, nms_score_list)


    def postprocess(self,*mxa_output):
        
        prior_bbox = np.squeeze(np.load(self.npy_file))
        
        bbox_logits_list = mxa_output[1][0]
        confidence_list = mxa_output[0][0]

        bbox_list = []
        score_list = []
        # bbox decode
        for index in range(int(len(prior_bbox[0]) / 4)):
            score = confidence_list[index * 2 + 0]
            if score < self.score_th:
                continue

            prior_x0 = prior_bbox[0][index * 4 + 0]
            prior_y0 = prior_bbox[0][index * 4 + 1]
            prior_x1 = prior_bbox[0][index * 4 + 2]
            prior_y1 = prior_bbox[0][index * 4 + 3]
            prior_cx = (prior_x0 + prior_x1) / 2.0
            prior_cy = (prior_y0 + prior_y1) / 2.0
            prior_w = prior_x1 - prior_x0
            prior_h = prior_y1 - prior_y0

            box_cx = bbox_logits_list[index * 4 + 0]
            box_cy = bbox_logits_list[index * 4 + 1]
            box_w = bbox_logits_list[index * 4 + 2]
            box_h = bbox_logits_list[index * 4 + 3]

            prior_variance = [0.1, 0.1, 0.2, 0.2]
            cx = prior_variance[0] * box_cx * prior_w + prior_cx
            cy = prior_variance[1] * box_cy * prior_h + prior_cy
            w = math.exp((box_w * prior_variance[2])) * prior_w
            h = math.exp((box_h * prior_variance[3])) * prior_h

            image_height, image_width = self.frame_height, self.frame_width
            bbox_list.append([
                int((cx - (w / 2.0)) * image_width),
                int((cy - (h / 2.0)) * image_height),
                int((cx - (w / 2.0)) * image_width) + int(w * image_width),
                int((cy - (h / 2.0)) * image_height) + int(h * image_height),
            ])
            score_list.append(float(score))
        # nmss
        keep_index = cv.dnn.NMSBoxes(
            bbox_list,
            score_list,
            score_threshold=self.score_th,
            nms_threshold=self.nms_th,
        )
        nms_bbox_list = []
        nms_score_list = []

        for index in keep_index:
            nms_bbox_list.append(bbox_list[index])
            nms_score_list.append(score_list[index])
        
        return nms_bbox_list, nms_score_list
        
    def draw(self,debug_image,nms_bbox_list, nms_score_list):
        # Draw bbox and score
        for bbox, score in zip(nms_bbox_list, nms_score_list):
            cv.putText(debug_image, '{:.3f}'.format(score), (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv.LINE_AA)
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)

        cv.putText(debug_image, "", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('vehicle detection', debug_image)
        self.out_video.write(debug_image)
        if cv.waitKey(1) == 27: 
            self.cam.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default='assets/road_traffic.mp4')
    args = parser.parse_args()

    cam = cv.VideoCapture(args.video)
    fps = cam.get(cv.CAP_PROP_FPS)
    app = App(cam, 256, fps)
    app.run_mxa()
