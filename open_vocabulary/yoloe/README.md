# Open-Vocab Segmentation YOLOE

The **YOLOE** example demonstrates **real-time open-vocabulary object detection and segmentation** using MemryX accelerators. This guide provides setup instructions, model details, and code snippets to help you quickly get started.

<p align="center">
    <img src="assets/yoloe.gif" alt="YOLOE Example" style="height: 300px;">
</p>

## Overview

<div style="display: flex">
<div style="">

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | YOLOE
| **Model Type**       | Open-Vocab Object Detection and Segmentation
| **Framework**        | ONNX
| **Model Source**     | [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/YoloE-v8s-seg_640_640_3_onnx.zip)
| **Input**            | Configurable (e.g., 640x640x3)
| **Output**           | Bounding boxes, segmentation masks, confidence scores
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)

## Requirements

Before running the application, ensure that **Python 3.8+**, **OpenCV**, and the **Ultralytics** library are installed. You can install the dependencies using the following commands:


```bash
pip3 install opencv-python==4.11.0.86 ultralytics
```

## Running the Application

### Step 1: Download and Prepare the Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:

```bash
wget https://developer.memryx.com/example_files/YoloE-v8s-seg_640_640_3_onnx.zip
mkdir -p models
unzip YoloE-v8s-seg_640_640_3_onnx.zip -d models
```


<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you need to compile the YOLOE model and generate the DFP file manually under models folder.

```bash
mkdir models 
cd models 
```

```Python
from ultralytics import YOLO

# Initialize a YOLOE model
model = YOLO("yoloe-v8s-seg.pt")

# Define custom classes
names = ["person"]
model.set_classes(names, model.get_text_pe(names))
model.export(format='onnx')
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yoloe-v8s-seg.onnx --model_in_out ../assets/boundary_map.json
```

</details>

### Step 2: Run the Program

With the compiled model, you can now run real-time inference. Below are examples of how to do this:


```bash
cd src/python
python3 demo.py  # default video path /dev/video0
python3 demo.py --input_video_path /dev/video0   # cam as input
python3 demo.py --input_video_path example.mp4   # video as input
```

The application supports the following video input formats: **MP4**, **AVI**, **MKV**, **MOV**, **WMV**, **FLV**, **WebM**, as well as live streams from a webcam.


## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/) 🔗  
  - License: [GPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE) 🔗


## Summary

This guide offers a quick and easy way to run open-vocabulary object detection and segmentation using the YOLOE model on MemryX accelerators. You can use the Python implementation to perform real-time inference. Clone the repository and follow the steps to get started immediately.

