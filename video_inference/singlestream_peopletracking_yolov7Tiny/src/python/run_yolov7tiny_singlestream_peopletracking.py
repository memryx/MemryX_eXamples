"""
Copyright (c) 2024 MemryX Inc.


GPL v3 License

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.

"""
# limit the numpy threads to improve performance
import os
import multiprocessing
os.environ["OMP_NUM_THREADS"] = str(max(1,int(multiprocessing.cpu_count()) // 4))

# Imports
import time
import argparse
import numpy as np
import cv2
from queue import Queue
from threading import Thread
from memryx import AsyncAccl
from yolov7 import YoloV7Tiny as YoloModel

###################################################################################################
###########################Tracking Code Start#####################################################
###################################################################################################

def linear_assignment(cost_matrix): #HungarianAlg
    from scipy.optimize import linear_sum_assignment
    x,y = linear_sum_assignment(cost_matrix)

    return np.array(list(zip(x,y)))

def iou_batch(bb_test, bb_gt):
    
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

    return(o)

def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    
    r = w / float(h)

    return np.array([x, y, s, r]).reshape((4, 1))

###################################################################################################
from copy import deepcopy
class KalmanFilter(object):
    def __init__(self):
        self.x = [[0],[0],[0],[0],[0],[0],[0]]     # np.zeros((7, 1)) # state

        self.F = np.array([[1,0,0,0,1,0,0],
                           [0,1,0,0,0,1,0],
                           [0,0,1,0,0,0,1],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1]])
        
        self.P = np.array([[10.,0.,0.,0.,0.,0.,0.],
                           [0.,10.,0.,0.,0.,0.,0.],
                           [0.,0.,10.,0.,0.,0.,0.],
                           [0.,0.,0.,10.,0.,0.,0.],
                           [0.,0.,0.,0.,10000.,0.,0.],
                           [0.,0.,0.,0.,0.,10000.,0.],
                           [0.,0.,0.,0.,0.,0.,10000.]])

        self.Q = np.array([[1.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,0.,0.,0.,0.,0.],
                           [0.,0.,1.,0.,0.,0.,0.],
                           [0.,0.,0.,1.,0.,0.,0.],
                           [0.,0.,0.,0.,0.5,0.,0.],
                           [0.,0.,0.,0.,0.,0.5,0.],
                           [0.,0.,0.,0.,0.,0.,0.25]])

        self.z = np.array([[None], # np.array([[None]*self.dim_z]).T
                           [None],
                           [None],
                           [None]]) 

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        if z is None:
            self.z = np.array([[None]*4]).T
            return

        R = np.array([[1.,0.,0.,0.],
                      [0.,1.,0.,0.],
                      [0.,0.,10.,0.],
                      [0.,0.,0.,10.]]) 

        H = np.array([[1,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0],
                      [0,0,1,0,0,0,0],
                      [0,0,0,1,0,0,0]])

        z = np.atleast_2d(z)

        y = z - np.dot(H, self.x)

        PHT = np.dot(self.P, H.T)

        S = np.dot(H, PHT) + R
        SI = np.linalg.inv(S)

        K = np.dot(PHT, SI)

        self.x = self.x + np.dot(K, y)

        I_KH = np.eye(7) - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, R), K.T)

        self.z = deepcopy(z)

###################################################################################################

class KalmanBoxTracker(object):
    
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter()
        self.kf.x[:4] = convert_bbox_to_z(bbox) # STATE VECTOR
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hit_streak = 0

        
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    
    def get_state(self):
        arr_u_dot = np.expand_dims(self.kf.x[4],0)
        arr_v_dot = np.expand_dims(self.kf.x[5],0)
        arr_s_dot = np.expand_dims(self.kf.x[6],0)

        return np.concatenate((convert_x_to_bbox(self.kf.x), [[0]], arr_u_dot, arr_v_dot, arr_s_dot), axis=1)
    

def associate_detections_to_trackers(detections, trackers):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > 0.2).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() ==1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]< 0.2):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self):
        self.trackers = []
        self.frame_count = 0
        
    def update(self, dets= np.empty((0,6))):
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)
        
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= 2 or self.frame_count <= 2):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) #+1'd because MOT benchmark requires positive value
            i -= 1

            if(trk.time_since_update > 5):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0,6))



###################################################################################################
#####################################Tracking Code End#############################################
###################################################################################################

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

###################################################################################################

class Yolo7Mxa:
    """
    A demo app to run YOLOv7 on the the MemryX MXA
    """

###################################################################################################
    def __init__(self, video_path, show = True, save = False):
        """
        The initialization function.
        """

        # Controls
        self.show = show
        self.save = save
        self.done = False

        # CV and Queues
        self.num_frames = 0
        self.cap_queue = Queue(maxsize=5)
        self.dets_queue = Queue(maxsize=5)
        if "/dev/video" in str(video_path):
            self.src_is_cam = True
        else:
            self.src_is_cam = False
        self.vidcap = cv2.VideoCapture(video_path, cv2.CAP_V4L2) 

        self.dims = ( int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )
        self.color_wheel = np.array(np.random.random([20,3])*255).astype(np.int32)

        # Model
        self.model = YoloModel(stream_img_size=(self.dims[1],self.dims[0],3))

        # Timing and FPS
        self.dt_index = 0
        self.frame_end_time = 0
        self.fps = 0
        self.dt_array = np.zeros([30])

        # Vedio writer
        if save:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.writer = cv2.VideoWriter('out.avi', fourcc, self.vidcap.get(cv2.CAP_PROP_FPS), self.dims)
        else:
            self.writer = None

        # Display and Save Thread
        # Runnting the display and save as a thread enhance the pipeline performance
        # Otherwise, the display_save method can be called from the output method
        self.display_save_thread = Thread(target=self.display_save,args=(), daemon=True)

        self.sort_tracker = Sort()
       

###################################################################################################
    def run(self):
        """
        The function that starts the inference on the MXA
        """

        # AsyncAccl
        accl = AsyncAccl(dfp='../../models/YOLO_v7_tiny_416_416_3_onnx.dfp')

        # Start the Display/Save thread
        print("YOLOv7-Tiny inference on MX3 started")
        self.display_save_thread.start()

        start_time = time.time()

        # Connect the input and output functions and let the accl run
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess)
        accl.wait()

        # Done
        self.done = True
        running_time = time.time()-start_time
        fps = self.num_frames / running_time
        print(f"Total running time {running_time:.1f}s for {self.num_frames} frames ... Average FPS: {fps:.1f}")

        # Wait for the Display/Save thread to exit
        self.display_save_thread.join()

###################################################################################################
    def capture_and_preprocess(self):
        """
        Captures a frame for the video device and pre-processes it.
        """
       
        while True:
            got_frame, frame = self.vidcap.read()

            if not got_frame:
                return None

            if self.src_is_cam and self.cap_queue.full():
                # drop frame and try again
                continue
            else:
                self.num_frames += 1
                
                # Put the frame in the cap_queue to be overlayed later
                self.cap_queue.put(frame)
                
                # Preprocess frame
                frame = self.model.preprocess(frame)
                return frame
        
###################################################################################################
    def postprocess(self, *mxa_output):
        """
        Post process the MXA output
        """

        # Post-process the MXA ouptut
        dets = self.model.postprocess(mxa_output)

        # Push the results to the queue to be used by the display_save thread
        self.dets_queue.put(dets)

        # Calculate current FPS
        self.dt_array[self.dt_index] = time.time() - self.frame_end_time
        self.dt_index +=1
        
        if self.dt_index % 15 == 0:
            self.fps = 1 / np.average(self.dt_array)

            if self.dt_index >= 30:
                self.dt_index = 0
        
        self.frame_end_time = time.time()


###################################################################################################
    def display_save(self):
        """
        Draws boxes over the original image. It will also conditionally display/save the image.
        """

        while self.done is False:
            
            # Get the frame from and the dets from the relevant queues
            frame = self.cap_queue.get()
            dets = self.dets_queue.get()
            self.cap_queue.task_done()
            self.dets_queue.task_done()

#-----------------------------------------------------------Tracking Code Start-----------------------------------------------------------------
            if len(dets):
                dets_to_sort = np.empty((0,6))

                for d in dets:
                    if d['class_idx'] == 0: #Only tracking people
                        dets_to_sort = np.vstack((dets_to_sort, np.array([d['bbox'][0],d['bbox'][1],d['bbox'][2],d['bbox'][3],d['score'],d['class_idx']])))
                
                tracked_dets = self.sort_tracker.update(dets_to_sort)

                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]

                    for i in range(len(tracked_dets)):
                        x1, y1, x2, y2 = int(bbox_xyxy[i][0]),int(bbox_xyxy[i][1]),int(bbox_xyxy[i][2]),int(bbox_xyxy[i][3])
                        cat = int(categories[i]) if categories is not None else 0
                        id = int(identities[i]) if identities is not None else 0
                        label = str(id) + ":"+ COCO_CLASSES[cat]
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,20), 2)
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
                        cv2.putText(frame, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, [255, 255, 255], 1)

            else:
                tracked_dets = self.sort_tracker.update()
#-----------------------------------------------------------Tracking Code End----------------------------------------------------------------------

            #if self.fps > 1:
            #    txt = f"{self.model.name} - {self.fps:.1f} FPS"
            #else:
            #    txt = f"{self.model.name}"
            #frame = cv2.putText(frame, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2) 

            # Show the frame
            if self.show:

                cv2.imshow('YOLOv7t Person Tracking on MX3', frame)

                # Exit on a key press
                if cv2.waitKey(1) == ord('q'):
                    self.done = True
                    cv2.destroyAllWindows()
                    self.vidcap.release()
                    exit(1)
            
            # Save the frame
            if self.save: 
                self.writer.write(frame)

###################################################################################################
###################################################################################################
###################################################################################################

def main(args):
    """
    The main funtion
    """

    yolo7_inf = Yolo7Mxa(video_path = args.video_path, show=args.show, save=args.save)
    yolo7_inf.run()

###################################################################################################

if __name__=="__main__":
    # The args parser
    parser = argparse.ArgumentParser(description = "\033[34mMemryX YoloV7-Tiny Demo\033[0m")
    parser.add_argument('--video_path', dest="video_path", 
                        action="store", 
                        default='/dev/video0',
                        help="the path to video file to run inference on. Use '/dev/video0' for a webcam. (Default: 'samples/soccer.mp4')")
    parser.add_argument('--save', dest="save", 
                        action="store_true", 
                        default=False,
                        help="The output video will be saved at out.avi")
    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    args = parser.parse_args()

    # Call the main function
    main(args)

# eof
