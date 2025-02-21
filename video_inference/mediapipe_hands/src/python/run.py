import os
import threading, json, cv2, sys, pickle
import numpy as np
from MxHandPose import MxHandPose

class MediapipeHandsDemo:
    def __init__(self, mxpose, **kwargs):

        self.mxpose          = mxpose

        # Create a threading event to signal threads to stop
        self.stop_event = threading.Event()
        self.camera_read_thread = threading.Thread(target=self.camera_read, daemon=True)
        self.display_thread = threading.Thread(target=self.and_display, daemon=True)

        self.cam_width = 0
        self.cam_height = 0
        self.cap = self.video_capture()

        # Start threads
        self.camera_read_thread.start()
        self.display_thread.start()

        # Keep the main thread alive
        self.camera_read_thread.join()
        self.display_thread.join()

    ##############################################################################################################
    # camera capture and display
    ##############################################################################################################

    def video_capture(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        # use these to force a resolution
        #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        
        self.cam_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return camera

    def camera_read(self):

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        while not self.stop_event.is_set():

            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Put the frame into mxpose's input queue if there's space
            if self.mxpose.full():
                # drop frame
                pass
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.mxpose.put(frame)


        self.mxpose.stop()
        self.cap.release()
        sys.exit(0)

    def and_display(self):

        cv2.namedWindow('MediapipeHands',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MediapipeHands', self.cam_width+20, self.cam_height+20)

        while not self.stop_event.is_set():

            # Check if the window is closed
            if cv2.getWindowProperty('MediapipeHands', cv2.WND_PROP_VISIBLE) < 1:
                break

            if not self.mxpose.empty():
                # .get() pulls from mxpose's output queue
                annotated_frame = self.mxpose.get()

                frame = self.draw(annotated_frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('MediapipeHands', frame)

                # Exit if 'q' or ESC is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27 is the ESC key
                    self.stop_event.set()  # Signal to stop threads
                    break

        cv2.destroyAllWindows()
        sys.exit(0)



    ##############################################################################################################
    # To Plot
    ##############################################################################################################

    def draw(self, annotated_frame):

        handkeypoints_lst, handtype_lst = self.get_handkeypoints_handtype(annotated_frame)
        img = annotated_frame.image
        for idx,handtype in enumerate(handtype_lst):
            img = self.drawLandmarks(img, [handkeypoints_lst[idx]], (handtype == "Left"))

        return img

    def get_handkeypoints_handtype(self, annotated_frame):
        handkeypoints_lst = []
        handtype_lst      = []
        for handpose in annotated_frame.handposes:
            hp_reshaped = handpose.landmarks.reshape(21, 3).astype(np.int32)
            singlehand_keypoints = [(int(x), int(y)) for x,y,z in hp_reshaped]
            handkeypoints_lst.append(singlehand_keypoints)
            handtype_lst.append(handpose.handedness)

        return handkeypoints_lst, handtype_lst

    def drawLandmarks(self, frame, data, is_left):
        allhands=data
        if is_left:
            color = (255,0,255)
        else:
            color = (0,255,255)
        for myHand in allhands:
            cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[1][0],myHand[1][1]),color,2)
            cv2.line(frame,(myHand[1][0],myHand[1][1]),(myHand[2][0],myHand[2][1]),color,2)
            cv2.line(frame,(myHand[2][0],myHand[2][1]),(myHand[3][0],myHand[3][1]),color,2)
            cv2.line(frame,(myHand[3][0],myHand[3][1]),(myHand[4][0],myHand[4][1]),color,2)
            cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[5][0],myHand[5][1]),color,2)
            cv2.line(frame,(myHand[5][0],myHand[5][1]),(myHand[6][0],myHand[6][1]),color,2)
            cv2.line(frame,(myHand[6][0],myHand[6][1]),(myHand[7][0],myHand[7][1]),color,2)
            cv2.line(frame,(myHand[7][0],myHand[7][1]),(myHand[8][0],myHand[8][1]),color,2)
            cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[17][0],myHand[17][1]),color,2)
            cv2.line(frame,(myHand[17][0],myHand[17][1]),(myHand[18][0],myHand[18][1]),color,2)
            cv2.line(frame,(myHand[18][0],myHand[18][1]),(myHand[19][0],myHand[19][1]),color,2)
            cv2.line(frame,(myHand[19][0],myHand[19][1]),(myHand[20][0],myHand[20][1]),color,2)
            cv2.line(frame,(myHand[5][0],myHand[5][1]),(myHand[9][0],myHand[9][1]),color,2)
            cv2.line(frame,(myHand[9][0],myHand[9][1]),(myHand[13][0],myHand[13][1]),color,2)
            cv2.line(frame,(myHand[13][0],myHand[13][1]),(myHand[17][0],myHand[17][1]),color,2)
            cv2.line(frame,(myHand[9][0],myHand[9][1]),(myHand[10][0],myHand[10][1]),color,2)
            cv2.line(frame,(myHand[10][0],myHand[10][1]),(myHand[11][0],myHand[11][1]),color,2)
            cv2.line(frame,(myHand[11][0],myHand[11][1]),(myHand[12][0],myHand[12][1]),color,2)
            cv2.line(frame,(myHand[13][0],myHand[13][1]),(myHand[14][0],myHand[14][1]),color,2)
            cv2.line(frame,(myHand[14][0],myHand[14][1]),(myHand[15][0],myHand[15][1]),color,2)
            cv2.line(frame,(myHand[15][0],myHand[15][1]),(myHand[16][0],myHand[16][1]),color,2)
            for i in myHand:
                cv2.circle(frame,(i[0],i[1]),4,(23,90,10),1)
            for i in myHand:
                cv2.circle(frame,(i[0],i[1]),3,(255,255,125),-1)
        return frame

if __name__ == '__main__':

    top_level_dir  = os.path.dirname(os.path.dirname(os.getcwd()))
    mx_modeldir    = os.path.join(top_level_dir, 'models')
    mx_pose        = MxHandPose(mx_modeldir=mx_modeldir, num_hands=2)
    paint          = MediapipeHandsDemo(mxpose=mx_pose)
