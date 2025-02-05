"""
Copyright (c) 2024 MemryX Inc.


GPLv3 or later License

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.

"""
import pygetwindow
import numpy as np
import cv2
import time
import pandas as pd
import dxcam
import queue
import win32api
import win32con
from threading import Thread

# local imports
from lib.accl import AsyncAccl
from lib.yolov7 import YoloV7Tiny


###################################################################################################
###################################################################################################
###################################################################################################

class AimbotMXA:

###################################################################################################
    def __init__(self):


        #######################
        #    <USER_CONFIG>    #
        #######################

        # size of screenshot region (centered on app window)
        self.screenShotHeight = 640
        self.screenShotWidth = 1600

        # right-shift the center of the screen,
        # might be useful for some games
        self.aaRightShift = 0

        # how rapidly to move the mouse towards a target [0,1]
        self.aaMovementAmp = 0.80

        # in-game key that will quit the aimbot
        self.aaQuitKey = "Q"

        # should we try to click too, or just aim?
        self.auto_click = True

        # aim higher in the box to try to get headshots
        self.headshot_mode = True

        # when there's multiple targets, shoot the one
        # closest to the center of the screen first
        self.centerOfScreen = True

        # show what the AI sees in a tiny box
        self.visuals = True

        #######################
        #    </USER_CONFIG>   #
        #######################




        # now don't touch anything below
        #-------------------------------

        #### vars
        # the screen capture obj
        self.camera = None

        # some vars
        self.cWidth = 0
        self.cHeight = 0
        self.center_screen = [0, 0]
        self.last_mid_coord = None
        self.clicked_recently = False

        # CV and Queues
        self.done = False
        self.cap_queue = queue.Queue(maxsize=4)
        self.dets_queue = queue.Queue(maxsize=4)
        self.dims = (self.screenShotWidth, self.screenShotHeight) # 1600x640
        self.color_wheel = np.array(np.random.random([20,3])*255).astype(np.int32)

        # contains the postproc
        self.model = YoloV7Tiny(stream_img_size=(self.dims[1],self.dims[0],3))

        # Timing and FPS
        self.num_frames = 0
        self.dt_index = 0
        self.frame_end_time = 0
        self.fps = 0
        self.dt_array = np.zeros([30])

        # post-proc thread
        self.display_shoot_thread = Thread(target=self.display_and_shoot, args=(), daemon=True)


###################################################################################################
    def init_capture(self):
        """
        Set up window capture for the user-selected application
        """

        # select the desired game window
        try:
            videoGameWindows = pygetwindow.getAllWindows()
            print("=== All Windows ===")
            for index, window in enumerate(videoGameWindows):
                # only list the window if it has a meaningful title
                if window.title != "":
                    print("[{}]: {}".format(index, window.title))
            # have the user select the window they want
            try:
                userInput = int(input(
                    "Please enter the number corresponding to the window you'd like to select: "))
            except ValueError:
                print("You didn't enter a valid number. Please try again.")
                return False
            # save that window as the chosen window for the rest of the script
            videoGameWindow = videoGameWindows[userInput]
        except Exception as e:
            print("Failed to select game window: {}".format(e))
            return False

        # activate window
        activationRetries = 30
        activationSuccess = False
        while (activationRetries > 0):
            try:
                videoGameWindow.activate()
                activationSuccess = True
                break
            except pygetwindow.PyGetWindowException as we:
                print("Failed to activate game window: {}".format(str(we)))
                print("Trying again... (you should switch to the game now)")
            except Exception as e:
                print("Failed to activate game window: {}".format(str(e)))
                print("Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
                activationSuccess = False
                activationRetries = 0
                break
            # wait a little bit before the next try
            time.sleep(3.0)
            activationRetries = activationRetries - 1

        # if we failed to activate the window then we'll be unable to send input to it
        # so just exit the script now
        if activationSuccess == False:
            return False

        print("Successfully activated the game window...")

        # set up the screen shots
        sctArea = {"mon": 1, "top": videoGameWindow.top + (videoGameWindow.height - self.screenShotHeight) // 2,
                             "left": self.aaRightShift + ((videoGameWindow.left + videoGameWindow.right) // 2) - (self.screenShotWidth // 2),
                             "width": self.screenShotWidth,
                             "height": self.screenShotHeight}

        # start screenshoting engine (dxcam)
        left = self.aaRightShift + \
            ((videoGameWindow.left + videoGameWindow.right) // 2) - (self.screenShotWidth // 2)
        top = videoGameWindow.top + \
            (videoGameWindow.height - self.screenShotHeight) // 2
        right, bottom = left + self.screenShotWidth, top + self.screenShotHeight

        region = (left, top, right, bottom)

        self.camera = dxcam.create(region=region, output_color="RGB", max_buffer_len=2)
        if self.camera is None:
            print("""DXCamera failed to initialize. Some common causes are:
            1. You are on a laptop with both an integrated GPU and discrete GPU. Go into Windows Graphic Settings, select python.exe and set it to Power Saving Mode.
             If that doesn't work, then read this: https://github.com/SerpentAI/D3DShot/wiki/Installation-Note:-Laptops
            2. The game is an exclusive full screen game. Set it to windowed mode.""")
            return False
        self.camera.start(target_fps=30, video_mode=True)

        # calculate the app window center
        self.cWidth = sctArea["width"] / 2
        self.cHeight = sctArea["height"] / 2
        self.center_screen = [self.cWidth, self.cHeight]

        return True


###################################################################################################
    def capture_and_preprocess(self):
        """
        Grabs new frames and gives them to the MXA (AsyncAccl's connected Input function)
        """

        if win32api.GetAsyncKeyState(ord(self.aaQuitKey)) != 0:
            return None

        while True:
            frame = np.array(self.camera.get_latest_frame())

            if self.cap_queue.full():
                # drop frame
                continue
            else:
                self.cap_queue.put(frame) # adds the original to a queue to display later
                frame = self.model.preprocess(frame) # preprocessing
                self.clicked_recently = False
                return frame # this goes into the MXA


###################################################################################################
    def postprocess(self, *mxa_output):
        """
        Gets outputs from the MXA, extracts detection data, and pushes to the dets queue
        """

        dets = self.model.postprocess(mxa_output)
        self.dets_queue.put(dets)


###################################################################################################
    def display_and_shoot(self):
        """
        Pops items off the dets queue, and uses it + the original image to shoot and display
        """

        while self.done is False:

            frame = self.cap_queue.get()
            self.cap_queue.task_done()

            # opencv likes to use BGR, not RGB, so let's flip the colors around
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

            dets = self.dets_queue.get()
            self.dets_queue.task_done()

            # collect targets from dets
            targets = []
            for d in dets:
                # we only care about the 'person' class
                #if d['class'] == "person":
                l,t,r,b = d['bbox']
                width = r - l
                height = b - t
                midx = l + (width/2)
                midy = t + (height/2)
                # save this info for later
                targets.append([midx, midy, width, height])

                # draw the bbox on the image (if visuals enabled)
                if self.visuals:
                    color = tuple([int(c) for c in self.color_wheel[d['class_idx']%20]])
                    frame = cv2.rectangle(frame, (l,t), (r,b), color, 2)
                    frame = cv2.rectangle(frame, (l,t-18), (r,t), color, -1)
                    frame = cv2.putText(frame, d['class'], (l+2,t-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # make targets a pandas frame for faster search ops, etc., later
            targets = pd.DataFrame(
                targets, columns=['current_mid_x', 'current_mid_y', 'width', 'height'])


            if len(targets) > 0:
                # find targets closest to the center (if this option is enabled)
                if self.centerOfScreen:
                    targets["dist_from_center"] = np.sqrt((targets.current_mid_x - self.center_screen[0])**2 + (targets.current_mid_y - self.center_screen[1])**2)

                    # sort by distance from center
                    targets = targets.sort_values("dist_from_center")

                # get the last person's mid coordinate if it exists
                if self.last_mid_coord:
                    targets['last_mid_x'] = self.last_mid_coord[0]
                    targets['last_mid_y'] = self.last_mid_coord[1]
                    # distance between current person mid coordinate and last person mid coordinate
                    targets['dist'] = np.linalg.norm(
                        targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                    targets.sort_values(by="dist", ascending=False)

                # take the first person that shows up in the dataframe (i.e., the closest person)
                xMid = targets.iloc[0].current_mid_x + self.aaRightShift
                yMid = targets.iloc[0].current_mid_y

                box_height = targets.iloc[0].height
                if self.headshot_mode:
                    headshot_offset = box_height * 0.40
                else:
                    headshot_offset = box_height * 0.10

                mouseMove = [xMid - self.cWidth, (yMid - headshot_offset) - self.cHeight]

                # move the mouse
                if win32api.GetKeyState(0x14):
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(
                        mouseMove[0] * self.aaMovementAmp), int(mouseMove[1] * self.aaMovementAmp), 0, 0)

                    if self.auto_click:
                        # try to limit how quickly we click
                        if not self.clicked_recently:
                            if abs(mouseMove[0]) <= 2.5:
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,int(mouseMove[0]),int(mouseMove[1]),0,0)
                                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,int(mouseMove[0]),int(mouseMove[1]),0,0)
                                self.clicked_recently = True
                        else:
                            # we must have moved on to the next target
                            if abs(mouseMove[0]) >= 10.0 or abs(mouseMove[1]) >= 20.0:
                                self.clicked_recently = False


                self.last_mid_coord = [xMid, yMid]

            else:
                self.last_mid_coord = None


            if self.visuals:

                # scale down the image so it doesn't take up the whole screen
                frame = cv2.resize(frame, (int(self.screenShotWidth / 4), int(self.screenShotHeight / 4)), interpolation=cv2.INTER_AREA)

                # show the frame
                cv2.imshow('YoloV7 Aimbot on MX3', frame)
                if cv2.waitKey(1) == ord('q'):
                    self.done = True
                    cv2.destroyAllWindows()
                    self.camera.stop()
                    exit(0)


###################################################################################################
    def run(self):

        accl = AsyncAccl(dfp='../models/yolov7-tiny_320_800.dfp')

        # start postproc thread
        self.display_shoot_thread.start()

        # connect and start
        accl.connect_output(self.postprocess)
        accl.connect_input(self.capture_and_preprocess)

        # run and wait for exit
        accl.wait()

        # done
        self.done = True

        # join
        self.display_shoot_thread.join()


###################################################################################################
###################################################################################################
###################################################################################################


def main():

    m = AimbotMXA()

    if m.init_capture() == False:
        return

    m.run()


if __name__ == "__main__":
    main()
