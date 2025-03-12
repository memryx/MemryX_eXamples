import os
import threading, json, cv2, sys, joblib
import numpy as np
import extra
from collections import Counter
from MxHandPose import MxHandPose

class ASLDemo:
    def __init__(self, mxpose, gesture_clf_dir, **kwargs):

        self.mxpose          = mxpose

        # Create a threading event to signal threads to stop
        self.stop_event = threading.Event()
        self.camera_read_thread = threading.Thread(target=self.camera_read, daemon=True)
        self.display_thread = threading.Thread(target=self.and_display, daemon=True)

        self.cam_width = 640
        self.cam_height = 480
        self.cap = self.video_capture()

        gesture_clf_path      = os.path.join(gesture_clf_dir, 'gesture_clf.pkl')

        self.word = []
        self.preds = []
        self.current_pred = None

        # Get gesture classifier for ASL Sign prediction
        if os.path.exists(gesture_clf_path):
            self.gesture_clf = joblib.load(gesture_clf_path)
            print("Success")
        else:
            print('No gesture data found')
            exit(1)

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
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        

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

        cv2.namedWindow('ASL Alphabet with Mediapipe Hands',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ASL Alphabet with Mediapipe Hands', self.cam_width+20, self.cam_height+20)

        while not self.stop_event.is_set():

            # Check if the window is closed
            if cv2.getWindowProperty('ASL Alphabet with Mediapipe Hands', cv2.WND_PROP_VISIBLE) < 1:
                break


            if not self.mxpose.empty():
                # .get() pulls from mxpose's output queue
                annotated_frame = self.mxpose.get()

                frame = self.draw(annotated_frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                word = ''.join(self.word)
                cv2.rectangle(frame, (0, self.cam_height - 80), (self.cam_width, self.cam_height), (255,255,255), -1)
                cv2.putText(frame, word, (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, lineType=cv2.LINE_AA)
                cv2.imshow('ASL Alphabet with Mediapipe Hands', frame)

                # Clear the current word when C is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'): 
                    self.word.clear()
                    self.preds.clear()

                # Delete the last character of the current word when backspace is pressed
                elif key == 8: 
                    if self.word:
                        self.word.pop()
                        self.preds.clear()
                
                # Add a space to the generated text if the space bar is pressed
                elif key == 32: # 32 is the space bar
                    if self.word:
                        self.word.append(" ")
                        self.preds.clear()

                # Exit if 'q' or ESC is pressed
                elif key == ord('q') or key == 27:  # 27 is the ESC key
                    self.stop_event.set()  # Signal to stop threads
                    break

        cv2.destroyAllWindows()
        sys.exit(0)



    ##############################################################################################################
    # To Plot
    ##############################################################################################################

    def draw(self, annotated_frame):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        handkeypoints_lst, handtype_lst, sign_pred = self.get_handkeypoints_handtype_signpred(annotated_frame)
        self.current_pred = sign_pred
        img = annotated_frame.image
        for idx,handtype in enumerate(handtype_lst):
            img = self.drawLandmarks(img, [handkeypoints_lst[idx]], (handtype == "Left"))

        return img

    def get_handkeypoints_handtype_signpred(self, annotated_frame):
        handkeypoints_lst = []
        handtype_lst      = []
        signpred_lst      = []
        for handpose in annotated_frame.handposes:
            hp_reshaped = handpose.landmarks.reshape(21, 3).astype(np.int32)
            singlehand_keypoints = [(int(x), int(y)) for x,y,z in hp_reshaped]
            classes = extra.classes
            signpred = extra.predict_sign(singlehand_keypoints, gesture_clf=self.gesture_clf, classes=classes)
            self.preds.append(str(signpred))
            handkeypoints_lst.append(singlehand_keypoints)
            handtype_lst.append(handpose.handedness)
            signpred_lst.append(signpred)

        return handkeypoints_lst, handtype_lst, signpred_lst

    def drawLandmarks(self, frame, data, is_left):
        allhands=data
        if is_left:
            color = (255,0,255)
        else:
            color = (0,255,255)
        for myHand in allhands:
            # Draw detected keypoints and connections on each detected hand
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
            # Print the predicted ASL sign for the current frame
            cv2.putText(frame, str(self.current_pred[0]), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, lineType=cv2.LINE_AA)
            
            # Logic for timing the signs and interpretations
            if len(self.preds) == 80: # Wait until 80 predictions have been made (over a period of ~3 seconds)
                counts = Counter(self.preds)
                best_pred = counts.most_common(1)[0][0]
                self.word.append(best_pred)
                self.preds.clear()
        return frame

if __name__ == '__main__':
    top_level_dir  = os.path.dirname(os.path.dirname(os.getcwd()))
    gesture_dir    = os.path.join(top_level_dir, 'data')
    mx_modeldir    = os.path.join(top_level_dir, 'models')
    mx_pose        = MxHandPose(mx_modeldir=mx_modeldir, num_hands=1)
    paint          = ASLDemo(mxpose=mx_pose, gesture_clf_dir=gesture_dir)