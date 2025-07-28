# Intrusion Detection

The **intrusion detection** example demonstrates real-time detection of intruding objects for any desired Region of Interest in a video feed. This can be very useful in video surveillance. ByteTrack has been used for tracking and Yolov8m has been used of the detection of objects

<p align="center">
  <img src="assets/intrusion.gif" alt="Depth Estimation Example" width="35%" />
</p>

## Overview

<div style="display: flex">
<div style="">

| **Property**         | **Details**
|----------------------|------------------------------------------
| **Models**           | [Yolov8m](https://docs.ultralytics.com/models/yolov8/), [ByteTrack](https://github.com/ifzhang/ByteTrack?tab=readme-ov-file)
| **Model Type**       | Object Detection, Tracking
| **Framework**        | [Tflite](https://www.tensorflow.org/)
| **Model Source**     | [Download from ultralytics](https://docs.ultralytics.com/models/yolov8/)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/2p0/yolov8m_intrusion_detection.zip)
| **Input**            | 640x640x3
| **Output**           | Bounding boxes, confidence scores, Tracking Ids.
| **OS**               | Linux
| **License**          | [AGPL](LICENSE.md)

## Requirements

Before running the application, ensure that all **requirements** are installed.

```bash
pip3 install -r requirements.txt
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/2p0/yolov8m_intrusion_detection.zip
mkdir -p models
unzip yolov8m_intrusion_detection.zip -d models
```


### Step 2: Running the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python.

#### Python

Download the video from the link given below and copy it to the location `assets/surveillance.mp4`. The video resolution should be 1920x1080p for the best results.

To run the Python example for real-time depth estimation using MX3, simply execute the following command:

```bash
cd src/python
python intrusion_demo.py --input_path ../../assets/surveillance.mp4
```
You can specify the video path with the following options:

* `--input_path`: Path to the video file

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [YoloV8m Model from Ultralytics](https://docs.ultralytics.com/models/yolov8/) 🔗
  - License: [AGPL-3.0](https://github.com/ultralytics/ultralytics/tree/main?tab=AGPL-3.0-1-ov-file) 🔗

- **Code Reuse**: Some code components, including byte track, were sourced from [ByteTrack - github repository](https://github.com/ifzhang/ByteTrack?tab=readme-ov-file) 🔗
  - License: [MIT](https://github.com/ifzhang/ByteTrack?tab=readme-ov-file) 🔗

- **Surveillance video**:  The provided video is a cropped video of the source video, [BLK-HDPTZ12 Security Camera](https://www.youtube.com/watch?v=U7HRKjlXK-Y) 🔗

## Summary

This guide offers a quick and easy way to intrusion detecion using the Object-Detection model on MemryX accelerators. You can use the Python implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
