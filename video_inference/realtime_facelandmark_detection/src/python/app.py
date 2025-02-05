import cv2 as cv
import numpy as np
from fdlite import FaceDetectionMXA, FaceLandmarkMXA
from fdlite.render import Colors, landmarks_to_render_data, render_to_image
from PIL import Image
from memryx import AsyncAccl
from queue import Queue

class FaceApp(FaceDetectionMXA):
    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.capture_queue = Queue(maxsize=4)

    def generate_frame(self):
        # Get frame
        while True:
            ok, frame = self.cam.read()
            if not ok:
                print("EOF")
                return None
            if self.capture_queue.full():
                # drop frame
                pass
            else:
                out, padding = self._preprocess(frame)
                self.capture_queue.put((frame,padding))
                return out

    def process_face(self, *ofmaps):
        (_, padding) = self.capture_queue.get()
        dets, face_roi = self._postprocess(ofmaps, padding, (self.input_width, self.input_height))

        if len(dets) == 0:
            return None
        return face_roi

class LandmarkApp(FaceLandmarkMXA):
    def __init__(self, cam_size):
        super().__init__()
        self.capture_queue = Queue(maxsize=4)
        self.canvas_size = cam_size

    def process_landmark(self, *ofmaps):
        (landmark_frame, padding, roi) = self.capture_queue.get()
        out = self._postprocess(ofmaps, padding, self.canvas_size, roi)
        render_data = landmarks_to_render_data(out, [], landmark_color=Colors.PINK, thickness=3)

        # render and display landmarks (points only)
        output_image = render_to_image(render_data, Image.fromarray(landmark_frame))
        return output_image

    def show(self, output_image):
        cv.imshow('Face Detection and Landmarks', np.array(output_image))
        
        k = cv.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            cv.destroyAllWindows()
            exit(1)

class App:
    def __init__(self, cam):
        self.face_app = FaceApp(cam)
        self.landmark_app = LandmarkApp(cam_size=(int(cam.get(cv.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))))
        self.face_roi = None
        self.capture_queue = Queue(maxsize=4)

    def generate_frame_face(self):
        frame = self.face_app.generate_frame() # Pre-processed frame
        if frame is None:
            return None
        orig_frame = self.face_app.capture_queue.get() # Original Frame
        self.capture_queue.put(orig_frame)
        self.face_app.capture_queue.put(orig_frame)
        return frame

    def process_face(self, *ofmaps):
        self.face_roi = self.face_app.process_face(*ofmaps)
        return self.face_roi

    def generate_frame_landmark(self):
        #if self.face_roi is not None:
        frame_pad = self.capture_queue.get()
        landmark_input_image, landmark_pad = self.landmark_app._preprocess(frame_pad[0], self.face_roi) 
        self.landmark_app.capture_queue.put((frame_pad[0], landmark_pad, self.face_roi))
        return landmark_input_image
    
    def process_landmark(self, *ofmaps):
        out = self.landmark_app.process_landmark(*ofmaps)
        self.landmark_app.show(out)
        return out

def run_mxa(dfp):
    accl = AsyncAccl(dfp)
    accl.connect_input(app.generate_frame_face)
    accl.connect_input(app.generate_frame_landmark, 1)
    accl.connect_output(app.process_face)
    accl.connect_output(app.process_landmark, 1)
    accl.wait()

if __name__ == '__main__':
    cam = cv.VideoCapture(0)
    app = App(cam)
    dfp_path = "../../models/models.dfp"
    run_mxa(dfp_path)

