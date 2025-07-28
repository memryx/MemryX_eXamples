# Realtime Face Landmark Detection Application 
This example demonstrates integrating the Face Detection and Face Mesh Detection pipeline into a realtime application. Models used for Face Detection and Landmark Detection are sourced from [Mediapipe Solutions Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide).

<p align="center">
  <img src="assets/sample.gif" alt="Demo of application" width="450">
</p>

## Overview

| Property             | Details                                                                 
|----------------------|-------------------------------------------------------------------------
| **Model**            | [Face Detection](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf), [Face Landmarks](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
| **Model Type**       | Face Detection + Face Landmarks
| **Framework**        | [LiteRT](https://ai.google.dev/edge/litert)
| **Model Source**     | [Face Detection](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite), [Face Landmarks](https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite)
| **Output**           | 468 3D Landmarks + FaceFlag
| **OS**               | Linux
| **License**          | [MIT License](LICENSE.md)                                         

## Requirements
Before running the application, ensure that OpenCV is installed. You can install it using the following commands:

```bash
pip install opencv-python==4.11.0.86
```

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/2p0/facelandmark.zip
mkdir -p models
unzip facelandmark.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model manually </summary>
If you prefer, you can download and compile the models rather than using the precompiled model. Download the Face Detection and Face Landmark models:

```bash
mkdir models && cd models/
wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite
```

You can now use the MemryX Neural Compiler to compile the models and generate the DFP file required by the accelerator.

```bash
mx_nc -v -m blaze_face_short_range.tflite face_landmark.tflite --autocrop
```

The Neural Compiler will generate the DFP file for the two models titled `models.dfp`. It will also create a postprocessing cropped model of the blaze face model (`model_0_blaze_face_short_range_post.tflite`).

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any pre/post-processing models are cropped out.

</details>

### Step 2: Run the Script

Move to the `src/python` directory and run the script. This will by default try to attach to source `1` for its video source.

```bash
cd src/python/
python3 app.py 
```

## Third-Party Licenses

This project uses third-party software, models, and libraries. Face Detection and Face Landmark Integration is sourced from patlevin's face-detection-tflite package. Below are the details of the licenses for these dependencies:

- **Model 1**: [face_detection_short_range.tflite](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite) 🔗 
  - License: [Apache License 2.0](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf) 🔗

- **Model 2**: [face_landmark.tflite](https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite) 🔗 
  - License: [Apache License 2.0](https://github.com/patlevin/face-detection-tflite/blob/main/LICENSE) 🔗

- **Code and Post-Processing**: Some Inference and Post-Processing were sourced from: ['face-detection-tflite' Github Repository](https://github.com/patlevin/face-detection-tflite/) 🔗  
  - License: [MIT License](https://github.com/patlevin/face-detection-tflite/blob/main/LICENSE) 🔗

## Summary

This example integrates the Face Detection and Face Mesh Detection pipeline into a realtime application.
