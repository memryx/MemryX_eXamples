# Face Detection & Emotion Classification using MultiModel

The **Face Detection & Emotion Classification** example demonstrates real-time Face Detection & Emotion Classification inference using the pre-trained ```face_detection_short_range.tflite``` model for face detection and the ```mobilenet_7.h5``` model for emotion recognition on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align=center>
    <img src="assets/face_emotion.png" alt="Face Detection & Emotion Classification Example">
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Face Detection](https://github.com/patlevin/face-detection-tflite) and [Emotion Classification](https://github.com/av-savchenko/face-emotion-recognition)     |
| **Model Type**       | Face Detection and Emotion Classification                                                        |
| **Framework**        | [TensorFlow](https://www.tensorflow.org/)                                                   |
| **Model Source**     | [Download from GitHub for Face Detection](https://github.com/patlevin/face-detection-tflite/blob/main/fdlite/data/face_detection_short_range.tflite) and [Download from GitHub for Emotion Recognition](https://github.com/av-savchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/mobilenet_7.h5) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/face_emotion_detection.zip)                                         |
| **Model Resolution** | 128 x 128 (Face Detection)  and 224 x 224  (Emotion Recognition)                                      |
| **Output**           | Face Bounding boxes and Emotion classes |
| **OS**               | Linux |
| **License**          | [MIT](LICENSE.md)     |

## Requirements

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV using the following commands:

```bash
pip install opencv-python
```

For C++ applications, download the supported version of PyTorch [PyTorch Get Started Page](https://pytorch.org/get-started/locally/)

```bash 
This application requires the PyTorch C++ library to run.

Download Instructions:

    Please download the supported version of PyTorch by following these steps at PyTorch Get Started Page

    1. PyTorch Build: Select 'Stable (2.3.0)'

    2. Your OS: Choose between Linux, Mac, or Windows

    3. Package: Select 'LibTorch'

    4. Language: Choose 'C++'

    5. Compute Platform: Select 'CPU'

    6. Installation Command: Ensure to download the 'cxx11 ABI' version, not the pre-built version.

    After downloading, extract the .zip file and place the file in the cpp folder.

Also, change the 'cxx11 ABI' version in the file CMakeLists.txt at lines 11 to match the downloaded version.
```

Also ensure that all memx runtime plugins and utilities libs are installed. For more information on installation, please refer to DevHub pages such as [memx runtime libs installation page](https://developer.memryx.com/docs_dev/get_started/install_driver.html) , and [third party libs installation page](https://developer.memryx.com/docs_dev/tutorials/requirements/installation.html)

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/face_emotion_detection.zip
mkdir -p models
unzip face_emotion_detection.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained face_detection_short_range.tflite model from face-detection-tflite GitHub:

```bash
wget https://github.com/patlevin/face-detection-tflite/blob/main/fdlite/data/face_detection_short_range.tflite
```

Download the pre-trained mobilenet model from emotion recognition from GitHub:

```bash
wget https://github.com/av-savchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/mobilenet_7.h5
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m  face_detection_short_range.tflite mobilenet_7.h5 --autocrop
```

Output:
The MemryX compiler will generate two files:

* `models.dfp`: The DFP file for the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python and C++.

#### Python

To run the Python example for real-time Face Detection & Emotion Classification using MX3, follow these steps:

Simply execute the following command:

```bash
python run_face_emotion.py
```
Command-line Options:
You can specify the model path and DFP (Compiled Model) path using the following options:

* `-d` or `--dfp`:  Path to the compiled DFP file (default is models/models.dfp)

Example:
To run with a specific model and DFP file, use:

```bash
python run_face_emotion.py -d <dfp_path> 
```

If no arguments are provided, the script will use the default paths for the model and DFP.

#### C++

To run the C++ example for real-time Face Detection & Emotion Classification using MX3, follow these steps:

1. Build the project using CMake. From the project directory, execute:

```bash
mkdir build
cd build
cmake ..
make
```

2. Run the application.

You need to specify whether you want to use the camera or a video file as input.

* To run the application using the default DFP file and a camera as input, use the following command:

```bash
./face_emotion_classification --cam
```

* To run the application with a video file as input, use the following command, specifying the path to the video file:

```bash
./face_emotion_classification --video <video_path>
```

* To run with a specific DFP file, use:

```bash
./face_emotion_classification [--cam or --video <path> or --img <path>] [-d <dfp_path>]
```

## Tutorial

A more detailed tutorial with a complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/docs/tutorials/realtime_inf/realtime_multimodel.html)

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model 1**: [face_detection_short_range.tflite](https://github.com/patlevin/face-detection-tflite/) 🔗 
  - License: [MIT](https://github.com/patlevin/face-detection-tflite/blob/main/LICENSE) 🔗

- **Model 2**: [mobilenet_7.h5](https://github.com/av-savchenko/face-emotion-recognition) 🔗 
  - License: [Apache License 2.0](https://github.com/av-savchenko/face-emotion-recognition/blob/main/LICENSE)  🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their GitHub: [face_detection_short_range.tflite](https://github.com/patlevin/face-detection-tflite/) and  [mobilenet_7.h5](https://github.com/av-savchenko/face-emotion-recognition)
  - License: [MIT](https://github.com/patlevin/face-detection-tflite/blob/main/LICENSE) 🔗
  - License: [Apache License 2.0](https://github.com/av-savchenko/face-emotion-recognition/blob/main/LICENSE)  🔗

## Summary

This guide offers a quick and easy way to run Face Detection & Emotion Classification using the multimodel model on MemryX accelerators. You can use either the Python or C++ implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
