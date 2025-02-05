import os
import threading, json, cv2, sys, pickle
import numpy as np
from MxHandPose import MxHandPose

class Virtual_Painter:
    def __init__(self, frame_queue, settings, gesture_datapath, **kwargs):

        self.frame_queue          = frame_queue
        self.settings             = settings
        
        self.settings['fps']            = 30
        self.settings['keypoints']      = [0,4,5,8,9,12,13,16,17,20]
        self.settings['confidence']     = 25
        self.settings['color_swatches'] = {
                                            "red":[0,0,255],
                                            "orange":[0,153,255],
                                            "yellow":[0,255,255],
                                            "green":[0,255,0],
                                            "cyan":[255,255,0],
                                            "blue":[255,0,0],
                                            "purple":[255,0,123],
                                            "pink":[255,0,255],
                                            "black":[0,0,0],
                                            "white":[255,255,255]
                                        }
        
        self.settings['brush_size']     = [5,10,15,20,25,30]

        if os.path.exists(gesture_datapath):
            with open(gesture_datapath,'rb') as f:
                    self.gesturenames  = pickle.load(f)
                    self.knowngestures = pickle.load(f)
        else:
            print('No gesture data found')
            sys.exit()


        # inital_draw_settings
        self.drawState='Standby'
        self.color='white'
        self.brush_size=20


        self.prevcanvas = np.zeros([self.settings['window_height'],self.settings['window_width'],3],dtype=np.uint8)
        self.prevframe  = np.zeros([self.settings['window_height'],self.settings['window_width'],3],dtype=np.uint8)

        # Create a threading event to signal threads to stop
        self.stop_event = threading.Event()    
        self.video_read_display_thread = threading.Thread(target=self.camera_read_and_display, daemon=True)
        

        # Start threads
        self.video_read_display_thread.start()

        # Keep the main thread alive
        self.video_read_display_thread.join()

    ##############################################################################################################
    # camera capture and display
    ##############################################################################################################

    def start_threads(self):
    
        # Start threads
        self.video_read_thread.start()
        self.video_display_thread.start()

        # Keep the main thread alive
        self.video_read_thread.join()
        self.video_display_thread.join()

    def video_capture(self):
        # Capture source ( camera or video)

        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings['window_height'])
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  self.settings['window_width'])
        camera.set(cv2.CAP_PROP_FPS,  self.settings['fps'])

        return camera

    def camera_read_and_display(self):

        cap = self.video_capture()
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        cv2.namedWindow('OpenCV Paint',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('OpenCV Paint',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)  
        
        while not self.stop_event.is_set():

            # Check if the window is closed
            if cv2.getWindowProperty('OpenCV Paint', cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Put the frame into the input queue
            # frame_queue is actually a MxHandPose object and
            # the .put (and .full) operate on the input queue
            if self.frame_queue.full():
                # drop frame
                pass
            else:
                self.frame_queue.put(frame)
            

            canvas          = self.prevcanvas
            frame           = self.prevframe
            if not self.frame_queue.empty():
                # frame_queue is actually a MxHandPose object
                # and the .get() pulls from the output queue
                annotated_frame = self.frame_queue.get()           
                
                if annotated_frame.num_detections:
                    canvas = self.draw(annotated_frame, canvas)                    
                frame   = cv2.addWeighted(annotated_frame.image,.4,canvas,1,1)

            frame   = self.minimal_UI(frame)    
            self.prevframe = frame
            self.prevcanvas = canvas
            cv2.imshow('OpenCV Paint', frame)
            # cv2.imshow('Only canvas', canvas)
             
            # Exit if 'q' or ESC is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is the ESC key
                self.stop_event.set()  # Signal to stop threads
                break
            elif key == ord('c'):
                self.prevcanvas = self.clearcanvas()

        self.frame_queue.stop()
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(0)
        


    ##############################################################################################################
    # To Plot
    ##############################################################################################################

    def draw(self, annotated_frame, canvas):

        threshold = self.settings['confidence']
        handkeypoints_lst, handtype_lst = self.get_handkeypoints_handtype(annotated_frame)
        img = annotated_frame.image
        for idx,handtype in enumerate(handtype_lst):
            if handtype == self.settings['command_hand']:
                distMatrix = self.findDistances(handkeypoints_lst[idx])
                error,idx2 = self.findError(distMatrix)
                if error < threshold and idx!=-1:
                    self.drawState = self.gesturenames[idx2]
                else:
                    self.drawState = 'Standby'
                img = self.drawLandmarks(img, [handkeypoints_lst[idx]])
                break

        if self.settings['command_hand'] not in handtype_lst:
            self.drawState='Standby'

        for idx,handtype in enumerate(handtype_lst):
            if handtype == self.settings['brush_hand']:
                cv2.circle(img,(handkeypoints_lst[idx][8][0], handkeypoints_lst[idx][8][1]), self.brush_size, self.settings['color_swatches'][self.color],-1)

                #! color swatches logic
                if handkeypoints_lst[idx][8][1]<60:
                    if handkeypoints_lst[idx][8][0]>0 and handkeypoints_lst[idx][8][0]<self.settings['window_width']//10:
                        self.color='red'
                    elif handkeypoints_lst[idx][8][0]>self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<2*self.settings['window_width']//10:
                        self.color='orange'
                    elif handkeypoints_lst[idx][8][0]>2*self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<3*self.settings['window_width']//10:
                        self.color='yellow'
                    elif handkeypoints_lst[idx][8][0]>3*self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<4*self.settings['window_width']//10:
                        self.color='green'
                    elif handkeypoints_lst[idx][8][0]>4*self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<5*self.settings['window_width']//10:
                        self.color='cyan'
                    elif handkeypoints_lst[idx][8][0]>5*self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<6*self.settings['window_width']//10:
                        self.color='blue'
                    elif handkeypoints_lst[idx][8][0]>6*self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<7*self.settings['window_width']//10:
                        self.color='purple'
                    elif handkeypoints_lst[idx][8][0]>7*self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<8*self.settings['window_width']//10:
                        self.color='pink'
                    elif handkeypoints_lst[idx][8][0]>8*self.settings['window_width']//10 and handkeypoints_lst[idx][8][0]<9*self.settings['window_width']//10:
                        self.color='white'
                    else:
                        self.color='black'

                #! brush size logic
                if handkeypoints_lst[idx][8][0]>0 and handkeypoints_lst[idx][8][0]<60 and handkeypoints_lst[idx][8][1]>60 and handkeypoints_lst[idx][8][1]<self.settings['window_height']-60:
                    diff=(self.settings['window_height']-120)//6
                    if handkeypoints_lst[idx][8][1]>60 and handkeypoints_lst[idx][8][1]<60+diff:
                        self.brush_size=5
                    elif handkeypoints_lst[idx][8][1]>60+diff and handkeypoints_lst[idx][8][1]<60+2*diff:
                        self.brush_size=10
                    elif handkeypoints_lst[idx][8][1]>60+2*diff and handkeypoints_lst[idx][8][1]<60+3*diff:
                        self.brush_size=15
                    elif handkeypoints_lst[idx][8][1]>60+3*diff and handkeypoints_lst[idx][8][1]<60+4*diff:
                        self.brush_size=20
                    elif handkeypoints_lst[idx][8][1]>60+4*diff and handkeypoints_lst[idx][8][1]<60+5*diff:
                        self.brush_size=25
                    else:
                        self.brush_size=30
                #! paint logic
                if self.drawState=='Draw':
                    cv2.circle(canvas,(handkeypoints_lst[idx][8][0],handkeypoints_lst[idx][8][1]),self.brush_size,self.settings['color_swatches'][self.color],-1)
        
        return canvas

    def get_handkeypoints_handtype(self, annotated_frame):
        handkeypoints_lst = []
        handtype_lst      = []
        for handpose in annotated_frame.handposes:
            hp_reshaped = handpose.landmarks.reshape(21, 3).astype(np.int32)
            singlehand_keypoints = [(int(x), int(y)) for x,y,z in hp_reshaped]
            handkeypoints_lst.append(singlehand_keypoints)
            handtype_lst.append(handpose.handedness)

        return handkeypoints_lst, handtype_lst
    
    def findDistances(self, handData):
        distMatrix=np.zeros([len(handData),len(handData)],dtype=np.float32)
        palmSize=((handData[0][0]-handData[9][0])**2+(handData[0][1]-handData[9][1])**2)**.5
        for rows in range(0,len(handData)):
            for columns in range(0,len(handData)):
                distMatrix[rows][columns]=(((handData[rows][0]-handData[columns][0])**2+(handData[rows][1]-handData[columns][1])**2)**.5)/palmSize
        return distMatrix
    
    def findError(self, unknownMatrix):
        error=9999999
        idx=-1
        for i in range(len(self.knowngestures)):
            currenterror=0
            for rows in self.settings['keypoints']:
                for columns in self.settings['keypoints']:
                    currenterror+=abs(self.knowngestures[i][rows][columns]-unknownMatrix[rows][columns])
            if currenterror<error:
                error=currenterror
                idx=i
        return error,idx

    def drawLandmarks(self, frame, data):
        allhands=data
        for myHand in allhands:
            cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[1][0],myHand[1][1]),(255,0,255),2)
            cv2.line(frame,(myHand[1][0],myHand[1][1]),(myHand[2][0],myHand[2][1]),(255,0,255),2)
            cv2.line(frame,(myHand[2][0],myHand[2][1]),(myHand[3][0],myHand[3][1]),(255,0,255),2)
            cv2.line(frame,(myHand[3][0],myHand[3][1]),(myHand[4][0],myHand[4][1]),(255,0,255),2)
            cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[5][0],myHand[5][1]),(255,0,255),2)
            cv2.line(frame,(myHand[5][0],myHand[5][1]),(myHand[6][0],myHand[6][1]),(255,0,255),2)
            cv2.line(frame,(myHand[6][0],myHand[6][1]),(myHand[7][0],myHand[7][1]),(255,0,255),2)
            cv2.line(frame,(myHand[7][0],myHand[7][1]),(myHand[8][0],myHand[8][1]),(255,0,255),2)
            cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[17][0],myHand[17][1]),(255,0,255),2)
            cv2.line(frame,(myHand[17][0],myHand[17][1]),(myHand[18][0],myHand[18][1]),(255,0,255),2)
            cv2.line(frame,(myHand[18][0],myHand[18][1]),(myHand[19][0],myHand[19][1]),(255,0,255),2)
            cv2.line(frame,(myHand[19][0],myHand[19][1]),(myHand[20][0],myHand[20][1]),(255,0,255),2)
            cv2.line(frame,(myHand[5][0],myHand[5][1]),(myHand[9][0],myHand[9][1]),(255,0,255),2)
            cv2.line(frame,(myHand[9][0],myHand[9][1]),(myHand[13][0],myHand[13][1]),(255,0,255),2)
            cv2.line(frame,(myHand[13][0],myHand[13][1]),(myHand[17][0],myHand[17][1]),(255,0,255),2)
            cv2.line(frame,(myHand[9][0],myHand[9][1]),(myHand[10][0],myHand[10][1]),(255,0,255),2)
            cv2.line(frame,(myHand[10][0],myHand[10][1]),(myHand[11][0],myHand[11][1]),(255,0,255),2)
            cv2.line(frame,(myHand[11][0],myHand[11][1]),(myHand[12][0],myHand[12][1]),(255,0,255),2)
            cv2.line(frame,(myHand[13][0],myHand[13][1]),(myHand[14][0],myHand[14][1]),(255,0,255),2)
            cv2.line(frame,(myHand[14][0],myHand[14][1]),(myHand[15][0],myHand[15][1]),(255,0,255),2)
            cv2.line(frame,(myHand[15][0],myHand[15][1]),(myHand[16][0],myHand[16][1]),(255,0,255),2)
            for i in myHand:
                cv2.circle(frame,(i[0],i[1]),4,(23,90,10),1)
            for i in myHand:
                cv2.circle(frame,(i[0],i[1]),3,(255,255,125),-1)
        return frame

    def minimal_UI(self, frame):

        settings                                    = self.settings
        drawState                                   = self.drawState
        color                                       = self.color
        brush_size                                  = self.brush_size
        color_idx                                   = ['red','orange','yellow','green','cyan','blue','purple','pink','white','black']
        frameleft                                   = frame[60:settings['window_height']-60,:80]
        objectframeleft                             = np.zeros([settings['window_height']-120,80,3],dtype=np.uint8)
        frameleft                                   = cv2.addWeighted(frameleft,.1,objectframeleft,1,0)
        frame[60:settings['window_height']-60,:80]  = frameleft
        framebottom                                 = frame[settings['window_height']-60:,:]
        objectframebottom                           = np.zeros([60,settings['window_width'],3],dtype=np.uint8)
        framebottom                                 = cv2.addWeighted(framebottom,.1,objectframebottom,1,0)
        frame[settings['window_height']-60:,:]      = framebottom

        cv2.line(frame,(0,60),(settings['window_width'],60),(10,10,10),2)
        cntr = 0
        for x in range(0,settings['window_width'],settings['window_width']//10):
            pt1 = (x,0)
            pt2 = (x+settings['window_width'],0)
            pt4 = (x,60)
            pt3 = (x+settings['window_width'],60)
            cv2.fillPoly(frame,[np.array([pt1,pt2,pt3,pt4])],settings['color_swatches'][color_idx[cntr]])
            cntr += 1
        cntr=0

        for x in range((settings['window_height']-120)//6,settings['window_height']-60,(settings['window_height']-120)//6):
            cv2.circle(frame,(40,x),settings['brush_size'][cntr],(255,255,255),-1)
            cntr+=1

        cv2.line(frame,(80,60),(80,settings['window_height']-60),(10,10,10),1)
        cv2.line(frame,(0,settings['window_height']-60),(int(3.4*settings['window_width']//5),settings['window_height']-60),(10,10,10),1)
        cv2.putText(frame,f'{drawState}',(20,settings['window_height']-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.line(frame,(settings['window_width']//8,settings['window_height']-60),(settings['window_width']//8,settings['window_height']),(10,10,10),1)
        pt1 = (settings['window_width']//7,settings['window_height']-50)
        pt2 = (2*settings['window_width']//7,settings['window_height']-50)
        pt3 = (2*settings['window_width']//7,settings['window_height']-10)
        pt4 = (settings['window_width']//7,settings['window_height']-10)
        cv2.fillPoly(frame,[np.array([pt1,pt2,pt3,pt4])],settings['color_swatches'][color])

        if color=='black':
            cv2.putText(frame,f'Eraser',(int(1.22*settings['window_width']//7),settings['window_height']-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.line(frame,(int(1.8*settings['window_width']//6),settings['window_height']-60),(int(1.8*settings['window_width']//6),settings['window_height']),(10,10,10),1)

        if brush_size==30:
            cv2.circle(frame,(int(2*settings['window_width']//6),settings['window_height']-30),brush_size-4,(255,255,255),-1)
        else:
            cv2.circle(frame,(int(2*settings['window_width']//6),settings['window_height']-30),brush_size,(255,255,255),-1)

        cv2.line(frame,(int(2.2*settings['window_width']//6),settings['window_height']-60),(int(2.2*settings['window_width']//6),settings['window_height']),(10,10,10),1)
        cv2.putText(frame,f'C to clear',(int(4.25*settings['window_width']//6),settings['window_height']-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.line(frame,(int(3.4*settings['window_width']//5),settings['window_height']-60),(int(3.4*settings['window_width']//5),settings['window_height']),(10,10,10),1)
        cv2.putText(frame,f'Q to quit',(int(4.34*settings['window_width']//5),settings['window_height']-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.line(frame,(int(3.3*settings['window_width']//4),settings['window_height']-60),(int(3.3*settings['window_width']//4),settings['window_height']),(10,10,10),1)
        cv2.putText(frame,f'Eraser',(int(2.74*settings['window_width']//3),40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        return frame

    def clearcanvas(self):
        return np.zeros([self.settings['window_height'],self.settings['window_width'],3],dtype=np.uint8)

if __name__ == '__main__':


    virtualpainter_dir     = os.path.dirname(os.path.dirname(os.getcwd()))
    mx_modeldir            = os.path.join(virtualpainter_dir, 'assets')
    mx_pose                = MxHandPose(mx_modeldir=mx_modeldir, num_hands=2)
    gesture_datapath = os.path.join(mx_modeldir, 'gesture_data.pkl')


    with open(os.path.join(mx_modeldir, 'settings.json')) as f:
        settings = json.load(f)

    paint                = Virtual_Painter(frame_queue=mx_pose, settings=settings, gesture_datapath=gesture_datapath)




