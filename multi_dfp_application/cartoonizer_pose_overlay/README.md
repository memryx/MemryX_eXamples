# Multi-DFP Example: Overlaying Results with Cartoonizer and Pose Estimation

This **Multi-DFP** example demonstrates how two distinct DFPs—Cartoonizer and Pose Estimation—can be combined and run as a single application on MemryX accelerators. Moreover, it also allows users to interact with the display and dynamically select which DFP to visualize, with the results from the selected DFPs synchronized and displayed in the same output visualization.

[Cartoonizer](../../fun_projects/cartoonizer/README.md) and [Pose Estimation](../../video_inference/pose_estimation_yolov8/README.md) are available as separate examples for single-application use; refer to the individual examples for standalone usage. Likewise, see the [Side-by-Side Multi-DFP Example](../cartoonizer_pose/README.md) for non-overlaying Multi-DFP applications.

<p align="center">
  <img src="assets/overlay.png" alt="Cartoon-PoseEstimation Overlay Example">

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
| **Output**           | cartoonized version of the input image and pose landmark coordinates |
| **OS**               | Linux |
| **License**          | [AGPL](LICENSE.md)                                       |

## Requirements (Linux)

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV and the Ultralytics package (for YOLO models) using the following commands:

```bash
pip install opencv-python==4.11.0.86
pip install PyQt5==5.15.11
pip install ultralytics==8.3.161
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

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python.

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
* `--display_fps`: Sets the display frame rate (default: 30). You can increase this value for a faster display, but setting it too high may cause visual artifacts like blockiness. It should not exceed the FPS of both cartoonizer and pose estimation applications.

Example:
To run with a specific model and DFP file, use:

```bash
python run.py [--cam | --video <video_path>] [--dfp_cartoon <cartoon_dfp_path>] [--dfp_pose <pose_dfp_path> --post <pose_post_processing_onnx_path>]
```

If no arguments are provided, the script will use the default paths for the model and DFP.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [From from GitHub](hhttps://github.com/SystemErrorWang/FacialCartoonization) 🔗 
  - License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) 🔗

- **Model**: [Yolov8s-pose from Ultralytics GitHub](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/ultralytics/ultralytics)  
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗
