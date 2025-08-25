# PCB Defect Detection Using YOLOv8

This example demonstrates **real-time, multi-stream PCB defect detection** using a custom-trained **YOLOv8** model on the **DsPCBSD** dataset, accelerated by **MemryX** hardware.  
Optimized C++ post-processing replaces ONNX Runtime for cropped-part inference, delivering higher performance and lower overhead on embedded systems.

This guide provides step-by-step setup instructions, model specifications, and practical code examples to help you quickly deploy and evaluate PCB defect detection using YOLOv8 on MemryX hardware.

<p align="center">
  <img src="assets/pcb_defect.png" alt="YOLOv8 PCB Defect Detection Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [YOLOv8s](https://docs.ultralytics.com/models/yolov8/)                  |
| **Model Type**       | Object Detection                                                        |
| **Framework**        | [ONNX](https://onnx.ai/)                                                |
| **Model Source**     | [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) (export to ONNX) |
| **Pre-compiled DFP** | [Download (ONNX)](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_small_640_640_3_onnx_DsPCBSD.zip) |
| **Model Resolution** | 640x640                                                                 |
| **Output**           | Bounding box coordinates with PCB defect class                          |
| **OS**               | Linux                                                                   |
| **License**          | [AGPL](LICENSE.md)                                                      |

## Requirements

Install the following before running the application:

1. **MemryX SDK**: [Get Started Guide](https://developer.memryx.com/get_started/)

2. **Build Essentials & Dependencies**
    ```bash
    sudo apt update
    sudo apt install -y cmake libopencv-dev qtbase5-dev qt5-qmake
    ```

## Running the Application

Follow these steps to set up and run the PCB defect detection demo:

### Step 1: Download DFP and Resources

```bash
cd mx_examples/optimized_video_monitor_apps/yolov8_pcb_defect_detection

# Download pre-compiled DFP model
wget https://developer.memryx.com/example_files/2p0/YOLO_v8_small_640_640_3_onnx_DsPCBSD.zip
mkdir -p models
unzip YOLO_v8_small_640_640_3_onnx_DsPCBSD.zip -d models

# Download resource files (videos, labels, etc.)
wget https://developer.memryx.com/example_files/DsPCBSD_resource.zip
mkdir -p assets
unzip DsPCBSD_resource.zip -d assets/
```

### Step 2: Build the Application

Compile the C++ application using CMake:

```bash
mkdir build
cd build
cmake ..
make -j4
```

### Step 3: Run the Demo

Supports multiple input sources (video files, USB cameras, IP cameras) and configurable DFP via a text file.

* Run with default configuration (4-channel video inference):

```bash
./yolov8_pcb_defect_detection -c ../assets/config.txt
```

> You can edit `assets/config.txt` to change input sources, channels, or other settings.

## Customizing `assets/config.txt`

Options include:

- Number of channels to process
- MemryX accelerator device selection
- Display screen index
- Input sources (video file, USB camera, RTSP IP camera)

**Example Format:**
```
dfp=models/YOLO_v8_small_640_640_3_onnx_DsPCBSD.dfp
num_chs=4
inf_confidence=0.3
inf_iou=0.45
group=0
screen_idx=0
video_predecoded_frames=500
video=assets/resource/video/DsPCBSD/video_01.mp4
usb_cam=0
ip_cam=rtsp://user:password@192.168.0.101/stream1
```
*Only valid key-value pairs are supported.*

## Third-Party Licenses

This project uses third-party software, models, and libraries:

- **Model**: [YOLOv8 from Ultralytics](https://docs.ultralytics.com/models/yolov8/)  
  License: [AGPL](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

- **Code & Pre/Post-Processing**: Portions from [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)  
  License: [AGPL](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

## Summary

Quickly deploy multi-stream PCB defect detection using YOLOv8 on MemryX accelerators.  
Download the code and pre-compiled DFP file to get started!
