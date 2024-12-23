import subprocess
import os
import time
import cv2 as cv
import numpy as np
import pyautogui
import sys
import signal
import threading
from queue import Queue
from mp_palmdet import MPPalmDet
from memryx import AsyncAccl
import argparse

# Disable the PyAutoGUI fail-safe feature
pyautogui.FAILSAFE = False

# Global variables
jump_cooldown = 5  # Frames to wait between consecutive jumps
cooldown_counter = 0  # Frame counter for jump cooldown
hand_detected = False  # To track the state of hand detection
chrome_process = None  # To track the Chrome process
display_frame = None  # Store frame for display in a separate thread
stop_display = False  # To signal when to stop the display thread

# Function to handle Ctrl+C (SIGINT)
def signal_handler(sig, frame):
    print("Exiting program with Ctrl+C...")
    cleanup_and_exit()

# Function to open Chrome browser
def open_chrome():
    global chrome_process
    chrome_path = "/usr/bin/google-chrome"  # Path to Chrome executable
    chrome_process = subprocess.Popen([chrome_path], start_new_session=True)  # Open Chrome browser
    print("Chrome opened!")

# Function to navigate to the Chrome Dinosaur Game
def start_dino_game():
    time.sleep(1)  # Wait for the browser to open
    pyautogui.typewrite('chrome://dino\n', interval=0.05)  # Navigate to Dinosaur game
    time.sleep(1)
    print("Game started!")

# Function to capture and preprocess the frame
def get_frame_and_preprocess():
    """
    Captures a frame from the camera, preprocesses it for the model, and returns it.
    """
    global display_frame
    got_frame, frame = cam.read()

    if not got_frame:
        print("Error: Could not capture frame from camera.")
        return None

    # Store the frame for the display thread
    display_frame = frame.copy()

    # Put the frame in the queue to be used later
    cap_queue.put(frame)

    # Preprocess the frame for the model
    return model._preprocess(frame)

# Helper function for reshaping and concatenating outputs
def reshape_and_concatenate(accl_output):
    reshaped_output0 = accl_output[0].reshape(1, 864, 1)
    reshaped_output1 = accl_output[1].reshape(1, 864, 18)
    reshaped_output2 = accl_output[2].reshape(1, 1152, 1)
    reshaped_output3 = accl_output[3].reshape(1, 1152, 18)
    
    accl_output_1 = np.concatenate((reshaped_output1, reshaped_output3), axis=1)
    accl_output_2 = np.concatenate((reshaped_output0, reshaped_output2), axis=1)
    
    return accl_output_1, accl_output_2

# Function to handle the post-processing and game control logic
def postprocess(*accl_output):
    global cooldown_counter, hand_detected
    
    frame = cap_queue.get()

    # Reshape and concatenate outputs from the accelerator
    accl_output_1, accl_output_2 = reshape_and_concatenate(accl_output)

    # Check if palm is detected
    is_hand_detected = model._postprocess(accl_output_1, accl_output_2)

    # Handle jump logic with cooldown
    if is_hand_detected and not hand_detected and cooldown_counter == 0:
        pyautogui.press('space')  # Simulate a jump in the game
        cooldown_counter = jump_cooldown  # Start cooldown
        hand_detected = True  # Update hand detection state
    elif not is_hand_detected:
        hand_detected = False  # Reset hand detection if no hand is detected

    # Decrease cooldown counter if it's active
    if cooldown_counter > 0:
        cooldown_counter -= 1

# Function to display video in a separate thread and keep the window on top
def display_video():
    global stop_display, display_frame
    while not stop_display:
        if display_frame is not None:
            # Show the video in a small window
            resized_frame = cv.resize(display_frame, (320, 240))  # Resize the frame to make it smaller
            cv.imshow('Video Feed', resized_frame)

            # Ensure the window stays on top (Linux with wmctrl)
            os.system('wmctrl -r "Video Feed" -b add,above')

            # Check if 'ESC' or 'q' key is pressed to exit
            key = cv.waitKey(1)
            if key == 27 or key == ord('q'):
                stop_display = True
                cleanup_and_exit()

# Cleanup and exit function to properly terminate resources
def cleanup_and_exit():
    """
    Releases all resources, closes the window, and exits the program.
    """
    global stop_display
    stop_display = True
    cv.destroyAllWindows()  # Close any OpenCV windows
    if cam.isOpened():
        cam.release()  # Release the camera resource if it was opened
    if chrome_process:
        chrome_process.terminate()  # Close Chrome if it's open
        print("Chrome closed.")
    print("Resources released. Exiting.")
    sys.exit(0)

# Main script execution
if __name__ == "__main__":
    # Argument parser for specifying DFP path
    parser = argparse.ArgumentParser(description="Run the Chrome Dinosaur game controlled by hand gestures.")
    parser.add_argument('-d', '--dfp', type=str, default="models/MediaPipe_palm_Detection_192_192_3_tflite.dfp",
                        help="Specify the path to the DFP model file. Default is 'models/MediaPipe_palm_Detection_192_192_3_tflite.dfp'.")
    
    args = parser.parse_args()

    # Handle Ctrl+C signal
    signal.signal(signal.SIGINT, signal_handler)

    # Open Chrome and start the Dinosaur Game
    open_chrome()
    start_dino_game()

    # Capture source (camera or video)
    src = sys.argv[1] if len(sys.argv) > 1 else '/dev/video0'
    cam = cv.VideoCapture(src)

    if not cam.isOpened():
        print(f"Error: Could not open video source {src}.")
        sys.exit(1)

    # Use the DFP file specified by the user or the default value
    dfp = args.dfp

    # Initialize model and queue
    model = MPPalmDet()
    cap_queue = Queue(maxsize=10)

    # Initialize the accelerator with the model
    accl = AsyncAccl(dfp=dfp)

    # Start the display thread
    display_thread = threading.Thread(target=display_video)
    display_thread.start()

    # Connect input and output functions to the accelerator
    accl.connect_input(get_frame_and_preprocess)
    accl.connect_output(postprocess)

    # Wait for the accelerator to process
    accl.wait()

    # Wait for display thread to finish
    display_thread.join()

# End of script
