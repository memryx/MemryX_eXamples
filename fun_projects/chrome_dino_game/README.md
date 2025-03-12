# Chrome Dinosaur Game Using MediaPipe Palm Detection Model

The **Chrome Dinosaur Game** example demonstrates how to control the Google Chrome "no internet" dinosaur game in real-time on the MX3 chip, utilizing MediaPipe’s palm detection model. By detecting hand gestures, you’ll be able to make the dinosaur jump by showing an open palm. This guide provides setup instructions, details about the model, and code snippets to help you quickly get started with gesture-based game control.

<p align="center">
  <img src="assets/dino_game.gif" alt="Dino Game Demo" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [MediaPipe Palm detection model](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)                                            |
| **Model Type**       | Chrome Dinosaur Game                                                        |
| **Framework**        | [TensorFlow](https://www.tensorflow.org/)                                                       |
| **Model Source**     | [Download from MediaPipe GitHub](https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/1p1/MediaPipe_palm_Detection_192_192_3_tflite.zip)                                           |
| **Model Resolution** | 192x192 (default)                                                       |
| **OS**               | Linux                                                       |
| **License**          | [MIT](LICENSE.md)                                     |

## Requirements

Before running the application, ensure that the following are installed:

* Python
* OpenCV
* PyAutoGUI
* wmctrl
* Google Chrome

You can install OpenCV and PyAutoGUI using the following commands:

```bash
pip install opencv-python
pip install pyautogui
```
Install tkinter on Linux to use MouseInfo:

```
sudo apt-get install python3-tk
```

To install wmctrl, use:

```bash
sudo apt-get install wmctrl
```
Follow these steps to install and launch Google Chrome:

1. Download the latest stable version of Google Chrome:
```
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
```

2. Install the downloaded package:
```
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

3. Launch Google Chrome:
```
google-chrome-stable
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget wget https://developer.memryx.com/model_explorer/1p1/MediaPipe_palm_Detection_192_192_3_tflite.zip
mkdir -p models
unzip MediaPipe_palm_Detection_192_192_3_tflite.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained MediaPipe Palm detection model from MediaPipe GitHub page:

```bash
wget https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite -O palm_detection_lite.tflite
```

### Step 2: Compile the Model (optional)

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m palm_detection_lite.tflite --autocrop -c 4
```

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now play the game in real-time. Below are examples of how to do this.

#### Python

To run the Python example and play the real-time Chrome Dinosaur Game using MX3, simply execute the following command:

```bash
cd src/python/
python run_dino_game.py
```
You can specify the model path and DFP (Compiled Model) path with the following option:

* `-d` or `--dfp`: Path to the compiled DFP file (default is models/MediaPipe_palm_Detection_192_192_3_tflite.dfp)

For example, to run with a specific DFP file, use:

```bash
python run_depth_estimate.py -d <dfp_path>
```

If no arguments are provided, the script will use the default DFP path.

## Tutorial

A more detailed tutorial with a complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/fun_projects/dino_game.html) 


## Third-Party License

This project uses third-party software and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [From Media_Pipe github](https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite) 🔗 
  - License: [Apache License 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) 🔗


## Summary

This guide provides a quick and easy way to play the Chrome Dinosaur Game using the MediaPipe Palm detection model on MemryX accelerators. With the Python implementation, you can play the game in real-time. Simply download the full code and the pre-compiled DFP file to get started immediately.
