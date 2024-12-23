"""
============
Information:
============
File Name: run_depth_estimate.py

============
Description:
============
A script to show how to use the Accelerator API to perform a real-time inference
on MX3. We will use a depth estimate model as our inference demo.
"""

###############################################################################
# Import necessary libraries ##################################################
###############################################################################

from os import system, path
import argparse
import cv2 as cv
import numpy as np
import sys
from memryx import AsyncAccl, NeuralCompiler

###############################################################################
# Parse command-line arguments ################################################
###############################################################################

# Parse command-line arguments for model path (-m) and DFP file (-d)
parser = argparse.ArgumentParser(description="Run MX3 real-time inference with options for model path and DFP file.")
parser.add_argument('-m', '--model', type=str, default="models/midas_v2_small.tflite", help="Specify the path to the model. Default is 'models/midas_v2_small.tflite'.")
parser.add_argument('-d', '--dfp', type=str, default="models/midas_v2_small.dfp", help="Specify the path to the compiled DFP file. Default is 'models/midas_v2_small.dfp'.")
args = parser.parse_args()

# Set model and DFP paths based on arguments
model_path = args.model
dfp_path = args.dfp

###############################################################################
# Download the Model ##########################################################
###############################################################################

# Download the model if necessary
if dfp_path and path.isfile(dfp_path):
    print("\033[93mCompiled DFP file found at {}. Skipping download.\033[0m".format(dfp_path))
elif path.isfile(model_path):
    print("\033[93mModel file found at {}. Skipping download.\033[0m".format(model_path))
else:
    print("\033[93mDownloading the model for the first time.\033[0m")
    
    # Download the tar.gz file
    system(f"curl -L -o ./midas_v2_small.tar.gz https://www.kaggle.com/api/v1/models/intel/midas/tfLite/v2-1-small-lite/1/download")
    
    # Extract the downloaded tar.gz file
    system(f"tar -xzf ./midas_v2_small.tar.gz -C ./")
    
    # Rename the extracted file (1.tflite) to model_path
    if path.isfile('./1.tflite'):
        system("mkdir -p models")
        system(f"mv ./1.tflite {model_path}")
        print("\033[93mModel extraction completed and renamed to {}.\033[0m".format(model_path))
    else:
        print("\033[91mError: Extracted file '1.tflite' not found.\033[0m")

###############################################################################
# Initializations #############################################################
###############################################################################

# Connect to the camera and get its properties
src = sys.argv[1] if len(sys.argv) > 1 else '/dev/video0'
cam = cv.VideoCapture(src)
input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))

###############################################################################
# Compile the Model if DFP is not provided ####################################
###############################################################################

# Compile the model if no precompiled DFP file is provided
if dfp_path and path.isfile(dfp_path):
    print("\033[93mUsing provided DFP file: {}.\033[0m".format(dfp_path))
    dfp = dfp_path
else:
    print("\033[93mCompiling the model for the first time. This step will be skipped if the compiled DFP exists.\033[0m")
    nc = NeuralCompiler(num_chips=4, models=model_path, verbose=1, dfp_fname="midas_v2_small")
    dfp = nc.run()

###############################################################################
# Input and Output functions ##################################################
###############################################################################

# Input function to capture and preprocess the frame
def get_frame_and_preprocess():
    """
    An input function for the accelerator to use. This input function will get
    a new frame from the cam and pre-process it.
    """
    got_frame, frame = cam.read()
    if not got_frame:
        return None

    # Pre-processing steps
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) / 255.0
    frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_CUBIC)
    frame = np.array(frame)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    frame = (frame - mean) / std

    return frame.astype("float32")

# Output function to process and display the result
def postprocess_and_show_frame(*accl_output):
    """
    An output function for the accelerator to use. This output function will
    post-process the accelerator output and display it on the screen.
    """
    prediction = accl_output[0]

    # Post-processing steps
    prediction = cv.resize(prediction, (input_width, input_height))
    depth_min = prediction.min()
    depth_max = prediction.max()
    postprocessed_output = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
    postprocessed_output = cv.applyColorMap(postprocessed_output, cv.COLORMAP_INFERNO)

    # Show the output
    cv.imshow('Depth Estimation using MX3', postprocessed_output)

    # Check if the window was closed
    if cv.getWindowProperty('Depth Estimation using MX3', cv.WND_PROP_VISIBLE) < 1:
        print("\033[93mWindow closed. Exiting.\033[0m")
        cv.destroyAllWindows()
        cam.release()
        exit(1)

    # Exit on a key press
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        cam.release()
        exit(1)

###############################################################################
# Accelerate ##################################################################
###############################################################################

print("\033[93mRunning Real-Time Depth Estimation\033[0m")
accl = AsyncAccl(dfp)
accl.connect_input(get_frame_and_preprocess)
accl.connect_output(postprocess_and_show_frame)
accl.wait()

# eof