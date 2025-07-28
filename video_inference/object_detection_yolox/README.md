# Object Detection Using YoloX-M model

The **Object Detection** example demonstrates real-time Object Detection inference using the pre-trained yoloX-M Object Detection model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/yolox.gif" alt="Object Detection Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [YoloX-M](https://github.com/Megvii-BaseDetection/YOLOX)                                            |
| **Model Type**       | Object Detection                                                        |
| **Framework**        | [ONNX](https://onnx.ai/)                                                   |
| **Model Source**     | [Download from YoloX GitHub](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/YOLOX_medium_640_640_3_onnx.zip)                                       |
| **Model Resolution** | 640x640                                                       |
| **Output**           | Total number of prediction grid points, along with bounding boxes, confidence scores, and class probabilities for detected objects (in anchor-free YOLOX, without predefined anchor boxes). |
| **OS**               | Linux |
| **License**          | [MIT](LICENSE.md)                                       |

## Requirements

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV using the following commands:

```bash
pip install opencv-python==4.11.0.86
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/YOLOX_medium_640_640_3_onnx.zip
mkdir -p models
unzip YOLOX_medium_640_640_3_onnx.zip -d models

cd models 
mv YOLOX_medium_640_640_3_onnx.dfp yolox_m.dfp
mv YOLOX_medium_640_640_3_onnx_post.onnx yolox_m_post.onnx

cd ..
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained YOLOX-M model and export it to ONNX:

```
mkdir models 
cd models 

wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yolox_m.onnx --autocrop 
```

Output:
The MemryX compiler will generate two files:

* `yolox_m.dfp`: The DFP file for the main section of the model.
* `yolox_m_post.onnx`: The ONNX file for the cropped post-processing section of the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python.

To run the Python example for real-time Object Detection using MX3, follow these steps:

Simply execute the following command:

```bash
python src/python/run_object_detection_yolox.py
```
Command-line Options:
You can specify the model path and DFP (Compiled Model) path using the following options:

* `-d` or `--dfp`:  Path to the compiled DFP file (default is models/yolox-m.dfp)
* `--video-source`: Path to video source or camera device (default is /dev/video0)

Example:
To run with a specific DFP file, and video sourse use:

```bash
python src/python/run_object_detection_yolox.py -d <dfp_path> --video-source <video_path>
```

If no arguments are provided, the script will use the default paths for the model and DFP.


## Third-Party License

This project uses third-party software and libraries. Below are the details of the licenses for these dependencies:

[Apache License 2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [YoloX GitHub](https://github.com/Megvii-BaseDetection/YOLOX) 🔗 
  - License: [Apache License 2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/Megvii-BaseDetection/YOLOX) 
  - License: [Apache License 2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) 🔗

- **Preview Image**: ["Mountain Skiing On Snow Covered Mountains" on Pexels](https://www.pexels.com/video/ski-montagne-skier-piste-de-ski-4274798/)  
  - License: [Pexels License](https://www.pexels.com/license/)

## Summary

This guide offers a quick and easy way to run Object Detection using the yoloX-M model on MemryX accelerators. You can use Python implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
