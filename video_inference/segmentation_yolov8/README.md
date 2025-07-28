# Segmentation Using Yolov8n-seg model

The **Segmentation** example demonstrates real-time Segmentation inference using the pre-trained yolov8n segmentation model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/segmentation.png" alt="Segmentation Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolov8n-seg](https://docs.ultralytics.com/models/yolov8/)                                            |
| **Model Type**       | Segmentation                                                        |
| **Framework**        | [onnx](https://onnx.ai/),[ tflite](https://www.tensorflow.org/)                                                  |
| **Model Source**     | [Download from Ultralytics GitHub or docs](https://docs.ultralytics.com/models/yolov8/) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/2p0/segmentation_yolov8.zip)                                           |
| **Model Resolution** | 640x640                                                       |
| **Output**           | Bounding boxes for detected objects, confidence scores, class labels, and segmentation masks |
| **OS**               | Linux |
| **License**          | [AGPL](LICENSE.md)                                      |

## Requirements

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV and the Ultralytics package (for YOLO models) using the following commands:

```bash
pip install opencv-python==4.11.0.86
```

```bash
pip install ultralytics==8.3.161
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/2p0/segmentation_yolov8.zip
unzip segmentation_yolov8.zip
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained YOLOv8n-seg model and export it to ONNX / TFLite:

You can use the following code to download the pre-trained yolov8n-seg.pt model and export it to ONNX format:

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model

# Export the model
model.export(format="onnx")
```

Additionally, it is essential to simplify the ONNX model by removing unnecessary nodes to ensure smooth compilation. Use the following command:

```bash
python -m onnxsim yolov8n-seg.onnx yolov8n-seg.onnx
```

If you don't have `onnxsim` installed in your environment, use the following command:

```bash
pip install onnxsim
```

Finally, if you want to export to TFLite format, please execute command:
```bash 
onnx2tf -i yolov8n-seg.onnx -o yolov8n-seg.tflite
```

If you need to install `onnx2tf`, use the following command to install it along with some additional dependencies:

```bash
pip install onnx2tf onnx_graphsurgeon ai_edge_libert sng4onnx
```
You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yolov8n-seg.onnx --autocrop -c 4
#or
mx_nc -v -m yolov8n-seg.tflite --autocrop -c 4
```

Output:
The MemryX compiler will generate two files:

* `yolov8n-seg.dfp`: The DFP file for the main section of the model.
* `yolov8n-seg_post.onnx`: The ONNX file for the cropped post-processing section of the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.

</details>


### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python and C++.

#### Python

To run the Python example for real-time Segmentation using MX3, follow these steps:

Simply execute the following command:

```bash
python src/python/run_segmentation.py
```
Command-line Options:
You can specify the model path and DFP (Compiled Model) path using the following options:

* `-d` or `--dfp`:  Path to the compiled DFP file (default is models/onnx/YOLO_v8_nano_seg_640_640_3_onnx.dfp)
* `-p` or `--post_model`: Path to the post-processing model file generated after compilation (default is models/onnx/YOLO_v8_nano_seg_640_640_3_onnx_post.onnx)

Example:
To run with a specific model and DFP file, use:

```bash
python src/python/run_segmentation.py -d <dfp_path> -p <post_processing_path>
```

If no arguments are provided, the script will use the default paths for the model and DFP.

#### C++

To run the C++ example for real-time segmentation using MX3, follow these steps:

1. Build the project using CMake. From the project directory, execute:

```bash
mkdir build
cd build
cmake ..
make
```

2. Run the application.

Simply execute the following command:

```bash
./segmentation_yolov8
```
Command-line Options:

* `--video`: Paths to video file as input (default is /dev/video0, camera connected to the system)
* `-d` or `--dfp`:  Path to the compiled DFP file (default is models/onnx/YOLO_v8_nano_seg_640_640_3_onnx.dfp)
* `-p` or `--post_model`: Path to the post-processing model file generated after compilation (default is models/onnx/YOLO_v8_nano_seg_640_640_3_onnx_post.onnx)


Example:
```bash
./segmentation_yolov8 --video <video_path> -d <dfp_path> -p <post_processing_path>
```

If no arguments are provided, the script will use camera as input, the default paths for the model and DFP.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Yolov8n-seg from Ultralytics GitHub](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/ultralytics/ultralytics)  
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Preview Image**: ["Man Sitting on a Seashore with His Husky Dog" on Pexels](https://www.pexels.com/photo/man-sitting-on-a-seashore-with-his-husky-dog-12461775/)  
  - License: [Pexels License](https://www.pexels.com/license/)

## Summary

This guide offers a quick and easy way to run Segmentation using the yolov8n-seg model on MemryX accelerators. You can use Python implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
