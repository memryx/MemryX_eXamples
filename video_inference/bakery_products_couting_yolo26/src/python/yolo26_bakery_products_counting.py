import cv2
import time
import argparse
import numpy as np
from queue import Queue
import supervision as sv
from threading import Thread
from memryx import AsyncAccl
from collections import defaultdict
from yolov26 import YoloV26 as YoloModel
from ultralytics.utils.plotting import colors


BAKERY_CONE_CLASSES = ("cone",)  # for yolo26 ice-cream/cone model
NEST_FILL_CLASSES = ("nest", )   # for yolo26  nest filling model


class Yolo26Mxa:
    """A demo app to run YOLOv26 on the MemryX MXA."""
    def __init__(self, video_path, dfp_path, postmodel_path, 
                line_coords : tuple[tuple[int, int], tuple[int, int]], show=True,
                output_path="test.mp4", save=False):
        """The initialization function."""

        # Controls
        self.show = show
        self.save = save
        self.done = False
        self.dfp_path = dfp_path
        self.postmodel_path = postmodel_path

        # Stream-related containers
        # CV and Queues
        self.num_frames = 0
        self.cap_queue = Queue(maxsize=4)
        self.dets_queue = Queue(maxsize=5)
        if "/dev/video" in str(video_path):
            self.src_is_cam = True
        else:
            self.src_is_cam = False
        self.vidcap = cv2.VideoCapture(video_path) 
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

        # Init bytetrack
        self.tracker = sv.ByteTrack()
        self.track_history = defaultdict(list)
        self.track_last_side = {}   # tid -> -1 or +1

        # Line Counter
        self.line_p1 = line_coords[0]
        self.line_p2 = line_coords[1]
        self.count_in = 0
        self.count_out = 0
        self.counted_ids = set()   # avoid double counting

        ### Constants
        self.bbox_thickness = 4
        self.text_scale = 2
        self.text_thickness = 5
        self.text_offset_y = 12
        self.text_padding = 15
        self.circle_radius = 8
        self.text_color = (255, 255, 255)  # White
        self.line_color = (108, 27, 255) # navyblue
        self.line_thickness = 5

        # Display and Save Thread
        # Runnting the display and save as a thread enhance the pipeline performance
        # Otherwise, the display_save method can be called from the output method
        self.display_thread = Thread(target=self.display,args=(), daemon=True)

        # Initialize video writer
        if self.save:
            fc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            self.vw = cv2.VideoWriter(output_path, fc, 
                                      self.vidcap.get(cv2.CAP_PROP_FPS),
                                      (self.dims[0],self.dims[1]))

    def run(self):
        """The function that starts the inference on the MXA."""
        print("dfp path = ", self.dfp_path)
        accl = AsyncAccl(dfp=self.dfp_path)
        print("YOLOv26 inference on MX3 started")
        accl.set_postprocessing_model(self.postmodel_path, model_idx=0)

        self.display_thread.start()

        start_time = time.time()

        # Connect the input and output functions and let the accl run
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess)
        accl.wait()
        self.done = True

        # Join the display thread
        self.display_thread.join()

    # Capture frames for streams and pre process
    def capture_and_preprocess(self):
        """Captures a frame for the video device and pre-processes it."""
        while True:
            got_frame, frame = self.vidcap.read()
            if not got_frame:
                return None
            if self.src_is_cam and self.cap_queue.full():
                # drop the frame and try again
                continue
            else:
                self.num_frames += 1
                
                # Put the frame in the cap_queue to be overlayed later
                self.cap_queue.put(frame)

                # OpenCV reads in BGR format, convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Preprocess frame for MXA and return it
                frame_rgb = self.model.preprocess(frame_rgb)
                return frame_rgb

    def crossed_line(self, prev_pt, curr_pt, eps=1e-6):
        s1 = self.side_of_line(prev_pt, self.line_p1, self.line_p2)
        s2 = self.side_of_line(curr_pt, self.line_p1, self.line_p2)
        if abs(s1) < eps or abs(s2) < eps:
            return False  # touching line ≠ crossing
        return s1 * s2 < 0

    def side_of_line(self, p, a, b):
        """
        Returns:
            >0 if point p is on one side
            <0 if on the other side
            =0 if exactly on the line
        """
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    
    def to_sv_detections(self, detections):
        if len(detections) == 0:
            return sv.Detections.empty()

        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidence = np.array([d["score"] for d in detections], dtype=np.float32)
        class_id = np.array([d["class_idx"] for d in detections], dtype=np.int64)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    
    def draw_counter_box(self, img, text, top_right, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        font, font_scale, thickness, padding =cv2.FONT_HERSHEY_SIMPLEX, 2.4, 7, 22
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x_right, y = top_right
        box_w = tw + padding * 2
        box_h = th + baseline + padding * 2
        box_x1 = x_right - box_w
        box_y1 = y
        box_x2 = x_right
        box_y2 = y + box_h

        # Draw background box
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
        center_x, center_y = (box_x1 + box_x2) // 2, (box_y1 + box_y2) // 2
        text_x, text_y = center_x - tw // 2, center_y + th // 2
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color,
                   thickness, cv2.LINE_AA)
        
    # Post process the output from MXA
    def postprocess(self, *mxa_output):
        """Post-process the MXA output."""
        
        dets = self.model.postprocess(mxa_output)  # Post-process the MXA ouptut
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

    # Display the output and show if opted in
    def display(self):
        """Continuously draws boxes + tracking + counting."""

        r = min(640 / self.dims[0], 640 / self.dims[1])

        new_w = int(self.dims[0] * r)
        new_h = int(self.dims[1] * r)

        pad_x = (640 - new_w) / 2
        pad_y = (640 - new_h) / 2

        while not self.done:

            frame = self.cap_queue.get()
            dets = self.dets_queue.get()
            self.cap_queue.task_done()
            self.dets_queue.task_done()

            tracks = self.tracker.update_with_detections(self.to_sv_detections(dets))

            for xyxy, cls, tid in zip(tracks.xyxy, tracks.class_id, tracks.tracker_id):
                print(cls)
                l, t, r_box, b = xyxy
                l -= pad_x
                r_box -= pad_x
                t -= pad_y
                b -= pad_y
                l /= r
                r_box /= r
                t /= r
                b /= r

                l, t, r_box, b = map(int, [l, t, r_box, b])

                l = max(0, min(l, self.dims[0] - 1))
                r_box = max(0, min(r_box, self.dims[0] - 1))
                t = max(0, min(t, self.dims[1] - 1))
                b = max(0, min(b, self.dims[1] - 1))

                color = colors(cls, True)

                x = int((l + r_box) / 2)
                y = int((t + b) / 2)

                track = self.track_history[tid]
                track.append((float(x), float(y)))

                if len(track) > 45:
                    track.pop(0)

                curr_pt = (x, y)
                side = self.side_of_line(curr_pt, self.line_p1, self.line_p2)

                DEAD_ZONE = 15
                if abs(side) < DEAD_ZONE:
                    continue

                side = 1 if side > 0 else -1

                if tid not in self.track_last_side:
                    self.track_last_side[tid] = side
                    continue

                if side != self.track_last_side[tid]:

                    dx = self.line_p2[0] - self.line_p1[0]
                    dy = self.line_p2[1] - self.line_p1[1]

                    nx, ny = -dy, dx

                    if len(track) >= 2:
                        mvx = track[-1][0] - track[-2][0]
                        mvy = track[-1][1] - track[-2][1]

                        direction = mvx * nx + mvy * ny

                        if direction > 0:
                            self.count_in += 1
                        else:
                            self.count_out += 1

                    self.counted_ids.add(tid)

                self.track_last_side[tid] = side

                cv2.rectangle(frame, (l, t), (r_box, b), color, self.bbox_thickness)
                label = BAKERY_CONE_CLASSES[int(cls)]

                (text_w, text_h) = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    self.text_thickness
                )[0]

                label_x1 = l
                label_y1 = max(0, t - text_h - self.text_padding * 2)
                label_x2 = l + text_w + self.text_padding * 2
                label_y2 = label_y1 + text_h + self.text_padding * 2

                cv2.rectangle(frame,
                            (label_x1, label_y1),
                            (label_x2, label_y2),
                            color, -1)

                text_y = label_y1 + text_h + self.text_padding // 2
                cv2.putText(frame,
                            label,
                            (l + self.text_padding, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.text_scale,
                            self.text_color,
                            self.text_thickness,
                            cv2.LINE_AA)

            cv2.line(frame, self.line_p1, self.line_p2, self.line_color, self.line_thickness)
            cv2.circle(frame, self.line_p1, self.circle_radius, self.line_color, -1)
            cv2.circle(frame, self.line_p2, self.circle_radius, self.line_color, -1)

            margin = 20

            self.draw_counter_box(
                frame,
                f"IN : {self.count_in}",
                top_right=(self.dims[0] - margin, margin),
                bg_color=(108, 27, 255)
            )

            self.draw_counter_box(
                frame,
                f"OUT: {self.count_out}",
                top_right=(self.dims[0] - margin, margin + 140),
                bg_color=(104, 31, 17)
            )

            if self.show:
                cv2.imshow("YOLOv26 on MemryX MXA", frame)

                if cv2.waitKey(1) == ord("q"):
                    self.done = True
                    cv2.destroyAllWindows()
                    self.vidcap.release()
                    exit(1)
            if self.save:
                self.vw.write(frame)


# Main function of the application
def main(args):
    """The main funtion"""
    yolo26_inf = Yolo26Mxa(video_path = args.video_path, dfp_path=args.dfp, 
                           postmodel_path=args.postmodel, show=args.show,
                           line_coords=((args.line_coords[0], args.line_coords[1]),
                             (args.line_coords[2], args.line_coords[3])),
                             save=args.save)
    yolo26_inf.run()

if __name__=="__main__":
 
    # The args parser to parse input paths
    parser = argparse.ArgumentParser(description="\033[34mRun MX3 real-time inference with options for DFP file and post model file path.\033[0m")
    
    parser.add_argument('-d', '--dfp', 
                        type=str, 
                        default="../../models/yolo26n-cone.dfp", 
                        help="Specify the path to the compiled DFP file. Default is 'models/yolo26n-cone.dfp'.")
    
    parser.add_argument('-m', '--postmodel', 
                        type=str, 
                        default="../../models/yolo26n-cone_post.onnx", 
                        help="Specify the path to the post-processing model. Default is 'models/yolo26n-cone_post.onnx'.")

    parser.add_argument('--video_path',  dest="video_path", 
                        action="store", 
                        default='path/to/video/file.mp4',
                        help="the path to video file to run inference on. Use '/dev/video0' for a webcam \n (Default: '/dev/video0')")

    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    parser.add_argument('-l', '--line_coords', type=int, nargs=4, default=[960, 0, 960, 1080],
                        help="Coordinates of the counting line in the order x1 y1 x2 y2. Default is 320 40 320 240")

    parser.add_argument('-s', '--save', action='store_true', help="Enable saving output to file. Output will be ./results.mp4  Default is False.")
    
    args = parser.parse_args()

    # Call the main function
    main(args)
