# 3D Point Cloud from Depth Estimation

The **Point Cloud from Depth Estimation** example demonstrates real-time depth inference using the pre-trained MiDaS v2 Small model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started with generating point clouds from depth data.

<p align="center">
  <img src="assets/point_cloud.gif" alt="Point-cloud Example" width="55%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [MiDaS v2 Small](https://arxiv.org/pdf/1907.01341) 🔗 
| **Model Type**       | Depth Estimation                                                        |
| **Framework**        | [LiteRT](https://ai.google.dev/edge/litert) 🔗
| **Model Source**     | [Download from TensorFlow Hub](https://www.kaggle.com/models/intel/midas) 🔗
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/MiDaS_256_256_3_tflite.zip)                                           
| **Input**            | 256x256 (default)                                                       
| **Output**           | Depth map (matches input resolution), Point Cloud Visualization         |
| **Application**      | Real-time point cloud generation and visualization from depth data      |
| **OS**               | Linux, Windows |
| **License**          | [MIT](LICENSE.md)                                         

## Requirements

### Linux

Before running the application, ensure that **OpenCV**, **Open3D**, and **curl** are installed. You can install the necessary libraries using the following commands:

```bash
pip install opencv-python==4.11.0.86 open3d==0.19.0
sudo apt install curl==8.5.0
```
for ARM (aarch64) platforms, pip install open3d is only supported for python 3.8 and 3.10.
### Windows

On Windows, first make sure you have installed [Python 3.11](https://apps.microsoft.com/detail/9nrwmjp3717k)🔗

Then open the `src/python_windows/` folder and double-click on `setup_env.bat`. The script will install all requirements automatically.


## Running the Application

### Step 1: Download Pre-compiled DFP

#### Windows

[Download](https://developer.memryx.com/model_explorer/2p0/MiDaS_256_256_3_tflite.zip) and place the .dfp file in the `python_windows/models/` folder.

#### Linux

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/MiDaS_256_256_3_tflite.zip
mkdir -p models
unzip MiDaS_256_256_3_tflite.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained MiDaS v2 Small model from TensorFlow Hub:

```bash
curl -L -o ./midas_v2_small.tar.gz https://www.kaggle.com/api/v1/models/intel/midas/tfLite/v2-1-small-lite/1/download
tar -xzf ./midas_v2_small.tar.gz -C ./
mkdir -p models
mv 1.tflite ./models/MiDaS_256_256_3_tflite.tflite
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -m models/MiDaS_256_256_3_tflite.tflite
```

</details>


### Step 2: Run the Script/Program

#### Linux
To run the Python example for real-time point cloud generation from depth data using MX3, simply execute the following command:

```bash
python src/python/run_pointcloud_from_depth.py
```
You can specify the model path and DFP (Compiled Model) path with the following options:

* `-m` or `--model`: Path to the model file (default is models/midas_v2_small.tflite)
* `-d` or `--dfp`: Path to the compiled DFP file (default is models/midas_v2_small.dfp)

For example, to run with a specific model and DFP file, use:

```bash
python src/python/run_pointcloud_from_depth.py -m <model_path> -d <dfp_path>
```

NOTE: In case you run into the following error:
```bash
[Open3D WARNING] GLFW Error: Wayland: The platform does not support setting the window position
[Open3D WARNING] Failed to initialize GLEW.
Running Real-Time Inference
[ WARN:0@5.219] global cap_v4l.cpp:803 requestBuffers VIDEOIO(V4L2:/dev/video0): failed VIDIOC_REQBUFS: errno=16 (Device or resource busy)
```

you can run the application by using the command below:

```bash 
XDG_SESSION_TYPE=x11 python src/python/run_pointcloud_from_depth.py
```

If no arguments are provided, the script will use the default model and DFP paths.

#### Windows

On Windows, you can instead just **double-click the `run.bat` file** instead of invoking the python interpreter on the command line.


### Interactive Point Cloud Visualization

The point cloud window is fully interactive, allowing you to pan, zoom, and tilt the view using your mouse:

- **Pan**: Left-click and drag to move the point cloud.
- **Zoom**: Scroll up or down with your mouse wheel to zoom in or out.
- **Tilt**: Right-click and drag to change the angle of the view.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [MiDaS v2 Small (TF Lite) from kaggle](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗 
  - License: [MIT](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from the MiDaS v2 Small model provided on [Kaggle](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1)  
  - License: [MIT](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗

## Summary

This guide offers a quick and easy way to generate point clouds from depth estimation using the MiDaS model on MemryX accelerators. You can use the Python implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
