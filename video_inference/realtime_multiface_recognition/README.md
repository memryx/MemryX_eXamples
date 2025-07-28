# Realtime Multi-Face Detection and Recognition Application 
This example demonstrates integrating the [Face Detection / Recognition](../../image_inference/face_recognition/README.md)
pipeline into a realtime application with an interactive GUI.  

<p align="center">
  <img src="assets/demo.gif" alt="Demo of application">
</p>


## Overview

| Property             | Details                                                                 
|----------------------|-------------------------------------------------------------------------
| **Model**            | [YoloV8n-Face](https://github.com/derronqi/yolov8-face), [FaceNet](https://arxiv.org/pdf/1503.03832)
| **Model Type**       | Face Detection + Recognition
| **Framework**        | [Onnx](https://onnx.ai/) + [Keras](https://keras.io/)
| **Model Source**     | [YoloV8n-Face](https://github.com/derronqi/yolov8-face), [FaceNet](https://github.com/serengil/deepface/blob/master/deepface/models/facial_recognition/Facenet.py)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/2p0/face_recognition.zip)                                           
| **Output**           | Face bounding box + keypoints + embedding
| **OS**               | Linux
| **License**          | [GPL](LICENSE.md)                                         

## Requirements

Before running the application, ensure that `OpenCV`, `PySide6`, and `lap` are installed. You can install them using the following commands:

```bash
pip install opencv-python==4.11.0.86
pip install PySide6==6.9.1
pip install lap==0.5.12
```
### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/2p0/face_recognition.zip
mkdir -p models
unzip face_recognition.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the models rather than using the precompiled model. Download the pre-trained yolov8n and FaceNet models:

```bash
wget https://developer.memryx.com/example_files/face_recognition_original.zip
unzip face_recognition_original.zip -d models
```

You can now use the MemryX Neural Compiler to compile the models and generate the DFP file required by the accelerator:

```bash
cd models/ 
mx_nc -v -m Facenet.h5 yolov8n-face_crop.onnx --dfp_fname yolov8n_facenet.dfp 
```

</details>

### Step 2: Run the Viewer

Move to the `src/` directory and run the viewer application. This will by default try to attach to `/dev/video0` for its video source.  

```bash
cd src/
python demo.py 
```

## Viewer Details

The viewer is an interactive GUI application developed using PyQT which
provides an interface to demonstrate realtime face recognition. Under the hood,
the MX-Accelerator is powering the face-recognition
pipeline. The Viewer will use a database to recognize the faces in the video stream. The databse can be preloaded or generated in realtime. New profiles can be added to the databse and images can be added to profiles by clicking on the faces in the video stream.
Some of the GUI features are highlighted below:

**Components**

- **Video source selector** (top-left): specify a path to the video to stream (defaults to `/dev/video0`). 
- **Detection Configuration Panel** (top-left): Select which parts of the face detection to visualize (boxes, keypoints, etc.).
- **Database Viewer** (left): Show the currently loaded database, with each profile and the images coresponding to each profile.
- **Image Previewer** (bottom-left): Show the currently highlighted image from the database viewer.

**Interactivity**

- Click on an `unknown` face to add a new `profile` for that user. 
- Click on a `recognized` face to capture more images for that `profile`.
- Highlight a `profile` (on the left) and click on any face to capture more images for that `profile`.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Models**: [Yolov8n-face Model exported from Yolov8-Face Github Repository](https://github.com/derronqi/yolov8-face) 🔗  
  - License: [GNU General Public License v3.0](https://github.com/derronqi/yolov8-face/blob/main/LICENSE) 🔗
- **Models**: [FaceNet Model exported from the DeepFace Github Repository](https://github.com/serengil/deepface) 🔗  
  - License: [MIT License](https://github.com/serengil/deepface/blob/master/LICENSE) 🔗

## Summary

This example integrates the `MXFace` Recognition pipeline into an end-to-end GUI application.
