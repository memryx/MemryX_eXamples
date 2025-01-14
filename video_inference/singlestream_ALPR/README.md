# Automatic License Plate Recognition (ALPR) Using multiple models

The **ALPR** example demonstrates multi-stream real-time object detection using the pre-trained LPD-Yunet model and pre-trained LPR-crnn model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [LPD-Yunet]([(https://github.com/opencv/opencv_zoo/tree/main/models/license_plate_detection_yunet)) and [LPR-crnn](https://github.com/we0091234/Chinese_license_plate_detection_recognition/tree/main)     |
| **Model Type**       | License Plate Detection and License Plate Recognition            |
| **Framework**        | [onnx](https://onnx.ai/)                                                    |
| **Model Source**     | [Download from GitHub for LPD-Yunet onnx model](https://github.com/opencv/opencv_zoo/tree/main/models/license_plate_detection_yunet), [Download from GitHub for LPR-crnn onnx model](https://github.com/we0091234/Chinese_license_plate_detection_recognition/tree/main)  |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/automatic_license_plate_recognition.zip)                                         |
| **Model Resolution** | 320x240 and 168x48                                              |
| **Output**           | License plate bounding box and license plate recognition result |
| **OS**               | Linux |
| **License**          | [GPL](LICENSE.md) |

## Requirements

For C++ applications, ensure that all memx runtime plugins and utilities libs are installed. For more information on installation, please refer to DevHub pages such as [memx runtime libs installation page](https://developer.memryx.com/get_started/install_driver.html) , and [third party libs installation page](https://developer.memryx.com/tutorials/requirements/installation.html)

```bash
sudo apt-get install memx-accl memx-accl-plugins memx-utils-gui 
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/automatic_license_plate_recognition.zip
mkdir -p models
unzip automatic_license_plate_recognition.zip -d models
```

### Step 2: Run the Script/Program


#### C++


To run the C++ example using MX3, follow these steps:


1. Build the project using CMake. From the project directory, execute:

```bash
mkdir build
cd build
cmake ..
make
```

2. Run the application. You can use single camera or provide video file for input.

* To run using the default DFP file and camera as input, simply run:

```bash
# ensure a camera device is connected as default video input is a cam
./singlestream_ALPR
```

* To run with a video file as input:

```bash
./singlestream_ALPR --video_paths vid:<path_to_video_file> 
```


## Third-Party Licenses

This project uses third-party software and models. Below are the details of the licenses for these dependencies:
 
* **English Model**: [Apache License 2.0](https://github.com/opencv/opencv_zoo/blob/main/models/license_plate_detection_yunet/LICENSE)
* **Chinese Model**: [GNU General Public License v3.0](https://github.com/we0091234/Chinese_license_plate_detection_recognition/blob/main/LICENSE)

## Summary

This guide offers a quick and easy way to run multi stream Automatic License Plate Recognition (ALPR) using LPD-Yunet and LPR-crnn model on MemryX accelerators. You can use C++ implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
