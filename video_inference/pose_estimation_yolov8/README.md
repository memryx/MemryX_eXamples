# Pose Estimation Using Yolov8m-pose model

The **Pose Estimation** example demonstrates real-time pose estimation inference using the pre-trained yolov8 medium pose model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/pose_estimation.gif" alt="Pose Estimation Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolov8m-pose](https://docs.ultralytics.com/models/yolov8/)                                            |
| **Model Type**       | Pose Estimation                                                        |
| **Framework**        | [ONNX](https://onnx.ai/)                                                   |
| **Model Source**     | [Download from Ultralytics GitHub or docs](https://docs.ultralytics.com/models/yolov8/) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/1p1/YOLO_v8_medium_pose_640_640_3_onnx.zip)                                          |
| **Model Resolution** | 640x640                                                       |
| **Output**           | Person bounding boxes and pose landmark coordinates |
| **OS**               | Linux, Windows |
| **License**          | [AGPL](LICENSE.md)                                       |

## Requirements (Linux)

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV and the Ultralytics package (for YOLO models) using the following commands:

```bash
pip install opencv-python
```

```bash
pip install ultralytics
```

For C++ applications, ensure that all memx runtime plugins and utilities libs are installed. For more information on installation, please refer to DevHub pages such as [memx runtime libs installation page](https://developer.memryx.com/get_started/install_driver.html) , and [third party libs installation page](https://developer.memryx.com/tutorials/requirements/installation.html)

## Running the Application (Linux)

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/1p1/YOLO_v8_medium_pose_640_640_3_onnx.zip
mkdir -p models
unzip YOLO_v8_medium_pose_640_640_3_onnx.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained YOLOv8m-pose model and export it to ONNX:

You can use the following code to download the pre-trained yolov8m-pose.pt model and export it to ONNX format:

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-pose.pt")  # load an official model

# Export the model
model.export(format="onnx")
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yolov8m-pose.onnx --autocrop -c 4
```

Output:
The MemryX compiler will generate two files:

* `yolov8m-pose.dfp`: The DFP file for the main section of the model.
* `yolov8m-pose_post.onnx`: The ONNX file for the cropped post-processing section of the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python and C++.

#### Python

To run the Python example for real-time Pose Estimation using MX3, follow these steps:

Simply execute the following command:

```bash
python run_pose_estimation.py
```
Command-line Options:
You can specify the model path and DFP (Compiled Model) path using the following options:

* `-d` or `--dfp`:  Path to the compiled DFP file (default is models/YOLO_v8_medium_pose_640_640_3_onnx.dfp)
* `-post` or `--post_model`: Path to the post-processing ONNX file generated after compilation (default is models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx)

Example:
To run with a specific model and DFP file, use:

```bash
python run_pose_estimation.py -d <dfp_path> -post <post_processing_onnx_path>
```

If no arguments are provided, the script will use the default paths for the model and DFP.

#### C++

To run the C++ example for real-time Pose Estimation using MX3, follow these steps:

1. Build the project using CMake. From the project directory, execute:

```bash
mkdir build
cd build
cmake ..
make
```

2. Run the application.

You need to specify whether you want to use the camera or a video file as input.

* To run the application using the default DFP file and a camera as input, use the following command:

```bash
./poseEstimation --cam
```

* To run the application with a video file as input, use the following command, specifying the path to the video file:

```bash
./poseEstimation --video <video_path>
```

## Running the Application (Windows)


[Download](https://developer.memryx.com/example_files/poseEstimation_windows.zip) the compiled C++ executable version, and extract the zip.

Then just double-click `run.bat` to launch using the first available webcam.

Alternatively, you can use commandline arguments such as `--video` if launching the exe within Command Prompt or PowerShell.


## Tutorial

A more detailed tutorial with a complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/realtime_inf/realtime_pose.html)

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Yolov8M-pose from Ultralytics GitHub](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/ultralytics/ultralytics)  
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

## Summary

This guide offers a quick and easy way to run Pose Estimation using the yolov8m-pose model on MemryX accelerators. You can use either the Python or C++ implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
