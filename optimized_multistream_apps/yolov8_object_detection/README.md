# Object Detection Using YOLOv8 (Optimized C++)

The **Object Detection** example demonstrates multi-stream (video / USB camera / IP camera) real-time object detection using the pre-trained YOLOv8 model on MemryX accelerators.  

In this example, instead of utilizing ONNX Runtime for running the cropped post-processing model, optimized C++ code performs the post-processing. This approach delivers higher performance and efficiency, especially on resource-constrained systems such as ARM platforms.  

This guide provides setup instructions, model details, and essential code snippets to help you get started quickly.


<p align="center">
  <img src="assets/result.png" alt="Yolov8n Object Detection Example" width="45%" />
</p>


## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [YOLOv8n](https://docs.ultralytics.com/models/yolov8/)                                            |
| **Model Type**       | Object Detection                                                      |
| **Framework**        | [ONNX](https://onnx.ai/)                                                   |
| **Model Source**     | [Download from Ultralytics GitHub or docs](https://docs.ultralytics.com/models/yolov8/) and export to onnx |
| **Pre-compiled DFP** | [Download here (ONNX)](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_nano_640_640_3_onnx.zip)   |
| **Model Resolution** | 640x640                                                 |
| **Output**           | Bounding box coordinates with objectness score, and class probabilities |
| **OS**               | Linux |
| **License**          | [AGPL](LICENSE.md)                                       |

## Requirements

Before running the application, make sure that the required packages are installed.

1. [MemryX Installation Guide](https://developer.memryx.com/get_started/install_runtime.html)

2. Installing Third-Party Packages
```bash
sudo apt install cmake libopencv-dev qtbase5-dev qt5-qmake
```

## Running the Application

### Step 1: Download Pre-compiled DFP and resources

```bash
cd mx_examples/optimized_video_monitor_apps/yolov8_object_detection
wget https://developer.memryx.com/model_explorer/2p0/YOLO_v8_nano_640_640_3_onnx.zip
mkdir -p models
unzip YOLO_v8_nano_640_640_3_onnx.zip -d models

wget https://developer.memryx.com/example_files/vms_resource.tar.gz
tar -xzvf vms_resource.tar.gz -C assets/
```

### Step 2: Run the Script/Program

With the compiled model, you can now build the C++ application and run real-time inference.

1. Build the project using CMake. From the project directory (mx_examples/optimized_video_monitor_apps/yolov8_object_detection), execute:

```bash
mkdir build
cd build
cmake ..
make -j 4
```

2. Run the application. The application supports multiple input sources, including cameras and video files, and allows you to specify a DFP using configuration text files.

* Run with the default configuration (4-channel video inference):

```bash
./yolov8_object_detection
```

This is equivalent to:

```bash
./yolov8_object_detection -c assets/config.txt
```

#### Customizing assets/config.txt

You can customize the assets/config.txt file to suit your needs, including the following options:

1. Number of channels to process.
2. YOLOv8 DFP model (supports [YOLOv8n](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_nano_640_640_3_onnx.zip), [YOLOv8s](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_small_640_640_3_onnx.zip), [YOLOv8m](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_medium_640_640_3_onnx.zip), etc.).
3. MemryX accelerator to use (if multiple devices are available).
4. Screen for display (if multiple screens are available).
5. Input sources (video file, USB camera, or RTSP IP camera).

* Configuration File Format (assets/config.txt). 

Ensure that the configuration file contains only valid key-value pairs without additional comments or formatting.

```
dfp=models/YOLO_v8_nano_640_640_3_onnx.dfp         [dfp path]
num_chs=4                                          [number of display channels]
inf_confidence=0.3                                 [yolov8 confidence level of detection]
inf_iou=0.45                                       [yolov8 iou threshold]
group=0                                            [group=0 means using /dev/memx0 memryx accelerator]
screen_idx=0                                       [display screen idx if you have many]
video_predecoded_frames=500                        [video file related setting]
video=resource/video/people_in_conference.mp4      [video file path]
video=resource/video/stop_sign.mp4
video=resource/video/walking_luggage.mp4
video=resource/video/meeting_room.mp4
usb_cam=0                                          [use /dev/video0 camera as example]
ip_cam=rtsp://user:password@192.168.0.101/stream1  [use rtsp ip-camera]
```


## Third-Party License

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Yolov8 from Ultralytics GitHub](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - License: [AGPL](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)  🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/ultralytics/ultralytics)  
  - License: [AGPL](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)  🔗
