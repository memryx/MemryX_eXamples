# YOLO OBB Object Detection

This example demonstrates **Object Detection** with *oriented boxes* using the
off-the-shelf YoloV8m-OBB model from Ultralytics compiled and running on the
MemryX accelerator. It implements an object detection pipeline for oriented
bounding boxes (OBB) where:

1. Objects are detected in an image.
2. Bounding boxes and keypoints are generated to represent detected objects.
3. The bounding boxes and keypoints are processed for further use in downstream tasks.

The implementation includes `MXObb`, which emulates a `Queue` structure, making
it easily integrable into a realtime application. The following demo code
utilizes `MXObb` to identify and count objects within an image.

![](assets/parking_lot_output.png)

## Overview

| Property             | Details                                                                                                                 |
|----------------------|-------------------------------------------------------------------------------------------------------------------------|
| **Model**            | [YoloV8m-OBB](https://github.com/ultralytics/ultralytics)                                                               |
| **Model Type**       | Object Detection (Oriented Bounding Boxes)                                                                              |
| **Framework**        | [Onnx](https://onnx.ai/)                                                                                                |
| **Model Source**     | [YoloV8m-OBB](https://github.com/ultralytics/ultralytics)                                                               |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/YOLO_v8_small_Oriented_Bounding_Boxes_1024_1024_3_onnx.zip) |
| **Output**           | Object bounding box + keypoints                                                                                         |
| **OS**               | Linux                                                                                                                   |
| **License**          | [AGPL](LICENSE.md)                                                                                                      |

## Requirements

Before running the application, ensure that ultralytics is installed:

```bash
pip install ultralytics==8.3.161
```

## Run the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
chmod +x models/download_model.sh; ./models/download_model.sh;
```

<details>
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled version. Download the pre-trained YoloV8m-OBB model:

```bash
cd models/
python3 export_model.py
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yolov8s-obb.onnx --autocrop 
```
</details>

### Step 2: Run the Script

With the compiled model, you can now use the MXA to perform object detection. Run the following script to see object detection in action:

```bash
cd src/
python run.py
```

By default, the script will download a sample image from google maps and perform object detection on them. Additionally, you can specify an image to detect objects in:

```bash
python run.py --image_path path/to/your/image.jpg
```

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Models**: [YoloV8m-OBB Model exported from the Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)  
  - License: [GNU Affero General Public License v3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **Default Image**: Default [parking lot image](https://www.pexels.com/photo/public-parking-with-modern-cars-in-rows-4196105/) 🔗  
  - License: [Pexels License](https://www.pexels.com/license/) 🔗

## Summary

This example implements an Object Detection with oriented boxes, utilizing the
off-the-shelf YoloV8m-OBB model. It showcases how to detect and localize
objects with oriented bounding boxes, making it a powerful tool for a variety
of detection tasks.


