# Multi-Stream Object Detection Using Yolov7 Tiny

The **Object Detection** example demonstrates real-time object detection using the pre-trained yolov7 tiny model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/yolov7_objectDetection_multistream.png" alt="MultiStream Object Detection Example" width="45%" />
</p>

For a single-stream input example, please refer to [single-stream object detection using yolov7 tiny](../../video_inference/singlestream_objectdetection_yolov7Tiny/README.md).

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolov7 Tiny](https://arxiv.org/pdf/2207.02696)                                            |
| **Model Type**       | Object Detection                                                      |
| **Framework**        | [onnx](https://onnx.ai/)                                                   |
| **Model Source**     | [Download](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt) and [export](https://github.com/WongKinYiu/yolov7/blob/main/export.py) to onnx |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/YOLO_v7_tiny_416_416_3_onnx.zip)                                           |
| **Dataset**          | [COCO](https://docs.ultralytics.com/datasets/detect/coco/) |
| **Model Resolution**            | 416x416                                                    |
| **Output**           | Bounding box coordinates with object probabilities |
| **OS**               | Linux |
| **License**          | [GPL](LICENSE.md)                                      |

## Requirements

Before running the application, ensure that Python and OpenCV are installed, especially for the Python implementation. You can install OpenCV using the following command:

```bash
# For application
pip install opencv-python==4.11.0.86

# For exporting source model to onnx
pip install pyyaml==6.0.2
pip install pandas==2.3.1
```
For C++ applications, ensure that all memx runtime plugins and utilities libs are installed. For more information on installation, please refer to DevHub pages such as [memx runtime libs installation page](https://developer.memryx.com/get_started/install_runtime.html) , and [third party libs installation page](https://developer.memryx.com/tutorials/requirements/installation.html)

```bash
sudo apt-get install memx-accl memx-accl-plugins memx-utils-gui 
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/YOLO_v7_tiny_416_416_3_onnx.zip
mkdir -p models
unzip YOLO_v7_tiny_416_416_3_onnx.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pretrained yolov7-tiny model from the source github.

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt -O yolov7tiny.pt

python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 416 416 --max-wh 416
```
The export script will generate a yolov7-tiny onnx file.

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator

```bash
 mx_nc -m yolov7-tiny.onnx -v --autocrop
```
The compiler will generate the DFP and a post-processing file which can be passed as inputs to the application.

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python and C++.

#### Python

To run the Python example for object detection with yolov7-tiny using MX3, simply execute the following command:

```bash
cd src/python/
# ensure a camera device is connected as default video input is a cam
python run_yolov7_multistream_objectdetection.py 
```
You can specify the model path and DFP (Compiled Model) path with the following options:

* `-m` or `--postmodel`: Path to the model file (default is models/YOLO_v7_tiny_416_416_3_onnx_post.onnx)
* `-d` or `--dfp`: Path to the compiled DFP file (default is models/YOLO_v7_tiny_416_416_3_onnx.dfp)

You can specify the input video path with the following option:

* `--video_paths` : Paths to video files as inputs (default is /dev/video0, camera connected to the system)

For example, to run with a specific video, post-processing model and DFP file, use:

```bash
python run_yolov7_multistream_objectdetection.py -m <postmodel_path> -d <dfp_path> --video_paths /dev/video0
```

You can specify multiple input video paths to run multiple stream with `--video_paths` option:

```bash
cd src/python/
python run_yolov7_multistream_objectdetection.py --video_paths /dev/video0 <video_path1> <video_path2>
```


If no arguments are provided, the script will use the default post-processing model and DFP paths.

#### C++

To run the C++ example using MX3, follow these steps:

1. Build the project using CMake. From the project directory, execute:

```bash
cd src/c++/

mkdir build
cd build
cmake ..
make
```

2. Run the application. You can use multiple cameras or provide video files for input, and specify a DFP file if needed.

* To run using the default DFP file and camera as input, simply run:

```bash
# ensure a camera device is connected as default video input is a cam
./multistream_objectdetection
```

* To run with a video file as input:

```bash
./multistream_objectdetection --video_paths vid:<path_to_video_file> 
```

* To specify a custom DFP file, use the `-d` option:


```bash
./multistream_objectdetection -d <path_to_dfp_file> 
```

* To specify multiple video inputs, use the `--video_paths` option with following format:

```bash
./multistream_objectdetection --video_paths vid:<video_path1>,vid:<video_path2>,cam:0
```

## Tutorial

A more detailed tutorial with complete code explanations is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/multistream_realtime_inf/multistream_od.html)


## Third-Party License

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Yolov7 Tiny from GitHub](https://github.com/WongKinYiu/yolov7) 🔗 
  - License: [GPL](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/WongKinYiu/yolov7)  
  - License: [GPL](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md) 🔗

## Summary

This guide offers a quick and easy way to run multi stream object detection using the Yolov7 Tiny model on MemryX accelerators. You can use either the Python or C++ implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
