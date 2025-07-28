# Face Detection + Face Recognition Pipeline
This example demonstrates **Face Detection / Recognition** utilizing  off-the-shelf YoloV8n-Face +
FaceNet models co-mapped and running on MemryX accelerator. It implements a
face recognition pipeline where:

1. Face boxes and keypoints are detected an image.
2. Faces are extracted from the image using bounding boxes and keypoints.
3. Extracted faces are embedded using the face recognition model.

The implementation includes `MXFace`, which emulates a `Queue` structure, making it easily integrable into a realtime application. The following demo code utilizes `MXFace` to distinguish the identity between two images.

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="assets/same.png" alt="Same Image" style="width: 100%;" />
    </td>
    <td style="text-align: center; border: none;">
      <img src="assets/different.png" alt="Different Image" style="width: 100%;" />
    </td>
  </tr>
</table>


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

Before running the application, ensure that OpenCV and Kagglehub re installed. You can install them using the following commands:

```bash
pip install opencv-python==4.11.0.86 kagglehub
```

## Run the Application 

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

### Step 2: Run the Script

With the compiled model, you can now use the MXA to perform face recognition. Run the following script to see face recognition in action:

```bash
cd src/
python3 run.py 
```

By default, the script will download a sample dataset of images from hit comedy T.V. series Friends and compare two sample images. Additionally, you can specify two images to distinguish:

```bash
python3 run.py --image_path path1 path2
```

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Models**: [Yolov8n-face Model exported from Yolov8-Face Github Repository](https://github.com/derronqi/yolov8-face) 🔗  
  - License: [GNU General Public License v3.0](https://github.com/derronqi/yolov8-face/blob/main/LICENSE) 🔗
- **Models**: [FaceNet Model exported from the DeepFace Github Repository](https://github.com/serengil/deepface) 🔗  
  - License: [MIT License](https://github.com/serengil/deepface/blob/master/LICENSE) 🔗

## Summary

This example impelments a Face Detection + Recognition pipeline, utilizing off-the shelf yolov8 + facenet models.
