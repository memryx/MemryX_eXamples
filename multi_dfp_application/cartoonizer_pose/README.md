# Multi-DFP Example Using Cartoonizer and Pose Estimation

The **Multi-DFP** example demonstrates how two distinct DFPs(Data Flow Pipelines) — Cartoonizer and Pose Estimation—can be combined and run as a single application on MemryX accelerators. It uses a pre-trained YOLOv8-small-pose model and an open-source cartoonizer model for real-time inference. 

While [Cartoonizer](../../fun_projects/cartoonizer/README.md) and [Pose Estimation](../../video_inference/pose_estimation_yolov8/README.md) are available as separate examples for single-application use, this example highlights how multiple applications can run together efficiently. Refer to the individual examples for standalone usage.

<p align="center">
  <img src="assets/cartoon_pose.png" alt="Cartoon-PoseEstimation Example">

</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Cartoonizer Model**    | [FacialCartoonization](https://github.com/SystemErrorWang/FacialCartoonization) |
| **Pose-Estimation model**  | [Yolov8s-pose](https://docs.ultralytics.com/models/yolov8/)                                            |
| **Model Type**       | Cartoonizer and Pose Estimation                                                        |
| **Framework**        | [ONNX](https://onnx.ai/)                                                   |
| **Model Source**     | [Cartoonizer](https://github.com/SystemErrorWang/FacialCartoonization/blob/master/weight.pth) and [Pose-Estimation](https://docs.ultralytics.com/models/yolov8/)|
| **Pre-compiled DFP** | [Cartoonizer](https://developer.memryx.com/model_explorer/2p0/Facial_cartoonizer_512_512_3_onnx.zip) and [Pose-Estimation](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_small_pose_640_640_3_onnx.zip)        |
| **Cartoonizer Model Resolution** | 512x512                         
| **Pose-Estimation Model Resolution** | 640x640                                                       |
| **Output**           | cartoonized version of the input image and Person bounding boxes and pose landmark coordinates |
| **OS**               | Linux |
| **License**          | [AGPL](LICENSE.md)                                       |

## Requirements (Linux)

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV and the Ultralytics package (for YOLO models) using the following commands:

```bash
pip install opencv-python==4.11.0.86
pip install PyQt5==5.15.11
pip install ultralytics==8.3.161
```

For C++ applications, ensure that all memx runtime plugins and utilities libs are installed. For more information on installation, please refer to DevHub pages such as [memx runtime libs installation page](https://developer.memryx.com/get_started/install_driver.html) , and [third party libs installation page](https://developer.memryx.com/tutorials/requirements/installation.html)

```bash
sudo apt-get install memx-accl memx-accl-plugins memx-utils-gui 
```

## Running the Application (Linux)

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
mkdir -p models
cd models

# Download and extract Facial Cartoonizer
wget https://developer.memryx.com/model_explorer/2p0/Facial_cartoonizer_512_512_3_onnx.zip
unzip Facial_cartoonizer_512_512_3_onnx.zip
rm Facial_cartoonizer_512_512_3_onnx.zip

# Download and extract YOLOv8 Small Pose
wget https://developer.memryx.com/model_explorer/2p0/YOLO_v8_small_pose_640_640_3_onnx.zip
unzip YOLO_v8_small_pose_640_640_3_onnx.zip
rm YOLO_v8_small_pose_640_640_3_onnx.zip

cd ..
```

<details> 

<summary> (Optional) Download and compile the model yourself </summary>

Download the pretrained model weights (weight.pth) from the **FacialCartoonization** GitHub repository

```bash
wget https://github.com/SystemErrorWang/FacialCartoonization/blob/master/weight.pth
```

Export the model to ONNX format. To help with the export process, you can refer to the generate_onnx.py script available in the zip folder, which shows you how to convert the model to ONNX format.

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m facial-cartoonizer_512.onnx --autocrop -c 4
```

Output:
The MemryX compiler will generate dfp file:

* `facial-cartoonizer_512.dfp`: The DFP file for the main section of the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.


Download the pre-trained **YOLOv8s-pose model** and export it to ONNX:

You can use the following code to download the pre-trained yolov8s-pose.pt model and export it to ONNX format:

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-pose.pt")  # load an official model

# Export the model
model.export(format="onnx")
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yolov8s-pose.onnx --autocrop -c 4 --dfp_fname YOLO_v8_small_pose_640_640_3_onnx
```

Output:
The MemryX compiler will generate two files:

* `yolov8s-pose.dfp`: The DFP file for the main section of the model.
* `yolov8s-pose_post.onnx`: The ONNX file for the cropped post-processing section of the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python and C++.

#### Python

To run the Python example using MX3, follow these steps:

Simply execute the following command:

```bash
cd src/python/
python run.py --cam
```
Run it with a video:

```bash
python run.py --video <video_path>
```

Command-line Options:
You can specify the model path and DFP (Compiled Model) path using the following options:

* `--dfp_cartoon`:  Path to the compiled DFP file of Cartoonizer (default is models/Facial_cartoonizer_512_512_3_onnx.dfp)
* `--dfp_pose`:  Path to the compiled DFP file of Pose Estimation (default is models/YOLO_v8_small_pose_640_640_3_onnx.dfp)
* `--post`: Path to the post-processing ONNX file generated after compilation (default is models/YOLO_v8_small_pose_640_640_3_onnx_post.onnx)

Example:
To run with a specific model and DFP file, use:

```bash
python run.py --cam/--video --dfp_cartoon <cartoon_dfp_path> --dfp_pose <pose_dfp_path> --post <pose_post_processing_onnx_path>
```

If no arguments are provided, the script will use the default paths for the model and DFP.

#### C++

To run the C++ example for using MX3, follow these steps:

1. Build the project using CMake. From the project directory, execute:

```bash
cd src/cpp/

mkdir build
cd build
cmake ..
make
```

2. Run the application.

You need to specify whether you want to use the camera or a video file as input.

* To run the application using the default DFP file and a camera as input, use the following command:

```bash
./cartoon_pose_app --cam
```

* To run the application with a video file as input, use the following command, specifying the path to the video file:

```bash
./cartoon_pose_app --video <video_path>
```

<!-- ## Tutorial

A more detailed tutorial with a complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/realtime_inf/realtime_pose.html) -->

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [From from GitHub](hhttps://github.com/SystemErrorWang/FacialCartoonization) 🔗 
  - License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) 🔗

- **Model**: [Yolov8s-pose from Ultralytics GitHub](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/ultralytics/ultralytics)  
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

## Summary

This guide offers a quick and easy way to run the Multi-DFP example, which demonstrates simultaneous real-time inference of Cartoonizer and Pose Estimation using MemryX accelerators. You can use the Python implementation to run both models in parallel with minimal setup. Download the full code along with the pre-compiled DFP files to get started immediately.
