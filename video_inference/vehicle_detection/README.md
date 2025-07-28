# Vehicle Detection

The **vehicle detection** example demonstrates real-time vehicle detection using the vehicle-detection-0200 model on MemryX accelerators. This guide provides setup instructions, model details, and code snippets to help you quickly get started.


<p align="center">
    <img src="assets/output.gif" alt="vehicle detection" style="height: 300px;">
</p>

## Overview

<div style="display: flex">
<div style="">

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | [vehicle_detection](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/178_vehicle-detection-0200)
| **Model Type**       | Object Detection
| **Framework**        | [Tflite](https://www.tensorflow.org/)
| **Model Source**     | [Download from PINTO](https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/178_vehicle-detection-0200/resources.tar.gz)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/Vehicle_Detection_0200_256_256_3_tflite.zip)
| **Input**            | 256x256x3
| **Output**           | Bounding boxes, confidence scores.
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)

## Requirements

Before running the application, ensure that **OpenCV** and **numpy** are installed, especially for the Python implementation. You can install OpenCV and curl using the following commands:

```bash
pip3 install opencv-python==4.11.0.86 numpy
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/Vehicle_Detection_0200_256_256_3_tflite.zip
mkdir -p models
unzip Vehicle_Detection_0200_256_256_3_tflite.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the model from github.

```bash
curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/178_vehicle-detection-0200/resources.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

```
This will create a folder containing multiple models. For this example, you should use the tflite model i.e. model_float32.tflite. Create the models folder using the commands below.

```bash
mkdir models 
cd models 
```

Once you have created this folder, move the **model_float32.tflite** from the resources folder into the newely created models folder and rename it to **Vehicle_Detection_0200_256_256_3_tflite.tflite**. 

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -m Vehicle_Detection_0200_256_256_3_tflite.tflite -v --autocrop
```
The compiler will generate the DFP, pre-processing and post-processing files which can be passed as inputs to the application.

</details>

### Step 2: Running the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python.

#### Python

To run the Python example for real-time depth estimation using MX3, simply execute the following command:

```bash
python3 src/python/run_vehicle_detection.py --video assets/road_traffic.mp4
```
You can specify the video path with the following options:

* `--video`: Path to the video file 

For example, to run with a specific model and DFP file, use:

```bash
python3 src/python/run_vehicle_detection.py --video <video_path>
```

If no arguments are provided, the script will use the default video paths.


## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Vehicle-Detection-0200 Model from the PINTO GitHub Repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/178_vehicle-detection-0200) 🔗  
  - License: [Apache-2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/178_vehicle-detection-0200/LICENSE) 🔗

- **Code Reuse**: Some code components, including pre/post-processing, were sourced from the demo code provided on [PINTO Model Zoo - vehicle_detection](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/178_vehicle-detection-0200/demo) 🔗  
  - License: [Apache-2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/178_vehicle-detection-0200/LICENSE) 🔗
"
- **Preview Video**: ["Traffic video" on Pexels](https://videos.pexels.com/video-files/2053100/2053100-sd_960_540_30fps.mp4) 🔗  
  - License: [Pexels License](https://www.pexels.com/license/) 🔗


## Summary

This guide offers a quick and easy way to run vehicle detection using the Vehicle-Detection model on MemryX accelerators. You can use the Python implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
