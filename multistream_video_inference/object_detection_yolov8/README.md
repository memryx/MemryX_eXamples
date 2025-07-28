# Object Detection Using yolov8s

The **Object Detection** example demonstrates multi-stream real-time object detection using the pre-trained yolov8s model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/yolov8_objectDetection.png" alt="Yolov8S Object Detection Example" width="45%" />
</p>


## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolov8s](https://docs.ultralytics.com/models/yolov8/)                                            |
| **Model Type**       | Object Detection                                                      |
| **Framework**        | [TensorFlow](https://www.tensorflow.org/) and [onnx](https://onnx.ai/)                                                   |
| **Model Source**     | [Download from Ultralytics GitHub or docs](https://docs.ultralytics.com/models/yolov8/) and export to tflite or onnx |
| **Pre-compiled DFP** | [Download here (TFLite)](https://developer.memryx.com/example_files/2p0/YOLO_v8_small_640_640_3_tflite.zip) or [Download here (ONNX)](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_small_640_640_3_onnx.zip)   |
| **Model Resolution** | 640x640                                                 |
| **Output**           | Bounding box coordinates with objectness score, and class probabilities |
| **OS**               | Linux |
| **License**          | [AGPL](LICENSE.md)                                       |

## Requirements

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV and the Ultralytics package (for YOLO models) using the following commands:

```bash
# For application
pip install opencv-python==4.11.0.86

# For exporting source model to onnx
pip install pyyaml==6.0.2
pip install pandas==2.3.1
```

```bash
pip install ultralytics==8.3.161
```

For C++ applications, ensure that all memx runtime plugins and utilities libs are installed. For more information on installation, please refer to DevHub pages such as [memx runtime libs installation page](https://developer.memryx.com/get_started/install_runtime.html) , and [third party libs installation page](https://developer.memryx.com/tutorials/requirements/installation.html)

```bash
sudo apt-get install memx-accl memx-accl-plugins memx-utils-gui 
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands: (Both tflite and ONNX is supported)
```bash
wget https://developer.memryx.com/example_files/2p0/YOLO_v8_small_640_640_3_tflite.zip
mkdir -p models/tflite
unzip -j YOLO_v8_small_640_640_3_tflite.zip -d models/tflite

wget https://developer.memryx.com/model_explorer/2p0/YOLO_v8_small_640_640_3_onnx.zip
mkdir -p models/onnx
unzip YOLO_v8_small_640_640_3_onnx.zip -d models/onnx
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained YOLOv8s model and export it to TFLite or ONNX:

You can use the following code to download the pre-trained yolov8s.pt model and export it to ONNX or TFLite format:

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load an official model

# Export the model

# TFLite format
model.export(format="tflite")   # creates 'yolo8s_float32.tflite' , use float32 model.

# ONNX format
model.export(format="onnx")
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yolov8s.tflite --autocrop -c 4
```
For ONNX:

```bash
 mx_nc -m yolov8s.onnx -v --autocrop -c 4
```
The compiler will generate the DFP file and a post-processing file, which can then be used as inputs to the application

</details>


### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python and C++.

#### Python

To run the Python example for object detection with yolov8s using MX3, simply execute the following command:

```bash
# ensure a camera device is connected as default video input is a cam
cd src/python/
python run_objectiondetection.py 
```
You can specify the model path and the DFP (Compiled Model) path using the following options. Both TFLite and ONNX formats are supported.

* `-p` or `--postmodel`: Path to the model file (default is models/tflite/YOLO_v8_small_640_640_3_tflite_post.tflite)
* `-d` or `--dfp`: Path to the compiled DFP file (default is models/tflite/YOLO_v8_small_640_640_3_tflite.dfp)

You can specify the input video path with the following option:

* `--video_paths` : Paths to video files as inputs (default is /dev/video0, camera connected to the system)

For example, to run with a specific video, post-processing model and DFP file, use:

```bash
python run_objectiondetection.py -p <postmodel_path> -d <dfp_path> --video_paths /dev/video0
```

You can specify multiple input video paths to run multiple stream with `--video_paths` option:

```bash
python run_objectiondetection.py --video_paths /dev/video0 <video_path1> <video_path2>
```

If no arguments are provided, the script will use the default post-processing model and DFP paths.

#### C++

To run the C++ example using MX3, follow these steps:

1. Build the project using CMake. From the project directory, execute:

```bash
cd src/cpp/

mkdir build
cd build
cmake ..
make
```

2. Run the application. You can use multiple cameras or provide video files for input, and specify a DFP file if needed.

* To run using the default DFP file and camera as input, simply run:

```bash
# ensure a camera device is connected as default video input is a cam
./multistream_objectdetection_yolov8s
```

* To run with a video file as input:

```bash
./multistream_objectdetection_yolov8s --video_paths vid:<path_to_video_file> 
```

* To specify a custom DFP file, use the `-d` option, and `-m` for post_model:


```bash
./multistream_objectdetection_yolov8s -d <path_to_dfp_file>  -m <path_to_post_model>
```

* To specify multiple video inputs, use the `--video_paths` option with following format:

```bash
./multistream_objectdetection_yolov8s --video_paths "vid:<video_path1>,vid:<video_path2>,cam:0"
```

## Tutorial

A more detailed tutorial with complete code explanations is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/multistream_realtime_inf/multistream_od_yolov8s.html)

## Third-Party License

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Yolov8 from Ultralytics GitHub](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - License: [AGPL](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)  🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/ultralytics/ultralytics)  
  - License: [AGPL](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)  🔗

## Summary

This guide offers a quick and easy way to run multi stream object detection using the Yolov8s model on MemryX accelerators. You can use either the Python or C++ implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
