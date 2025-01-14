# YOLOv8 Object Detection Accuracy Calculation

The **YOLOv8 Object Detection** example demonstrates how to validate the accuracy (mAP) of a pretrained YOLOv8 checkpoint on the COCO dataset using MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

## Overview

| Property             | Details                                                                              |
| -------------------- | ------------------------------------------------------------------------------------ |
| **Model**            | [YOLOv8](https://docs.ultralytics.com/models/yolov8/)                                |
| **Model Type**       | Object Detection                                                                     |
| **Framework**        | [PyTorch](https://pytorch.org/)                                                      |
| **Model Source**     | [Ultralytics](https://github.com/ultralytics/ultralytics) (Downloaded Automatically) |
| **Input**            | 640x640x3 (default)                                                                  |
| **Output**           | Bounding boxes and class probabilities                                               |
| **OS**               | Linux                                                                                |
| **License**          | [AGPL](LICENSE.md)                                                                   |

## Requirements

Before running the application, ensure that Python, and the `ultralytics` package are installed:

```bash
pip install ultralytics
```

## Running the Application

The application is contained in `src/run_validation.py`. It takes two optional arguments:
* `--size` or `-s` - Specifies the size of the YOLOv8 model (n, s, m). Default is 'm'. The 'l' and 'x' models will require 8 chips to run and is not supported here.
* `--device` or `-d` - Specifies the device to run the validation on (mxa, cpu). Default is 'mxa'. If 'cpu' is specified and a CUDA GPU is available, then the GPU is used.

The model (~50 MB) and COCO dataset (~20 GB) are downloaded automatically if not found locally. If a pre-compiled DFP and post-processing onnx model are not found, then the model is automatically compiled to generate these. Refer to the `compile_model` function to see how this is done. 

We have provided these files for `yolov8m` if you want to skip the compilation, other sizes will be compiled by the script and placed in the `weights/` directory:

```bash
wget https://developer.memryx.com/example_files/detection_accuracy_yolov8.zip
unzip detection_accuracy_yolov8.zip -d weights
```

To run the application:

```bash
python src/run_validation.py
```

Other ways to run:
```bash
python src/run_validation.py --device cpu   # Runs on cpu/gpu (establish baseline performance) 
python src/run_validation.py --size n       # Runs validation on the nano model
```

## Tutorial

A more detailed tutorial with a complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/accuracy/yolov8_accuracy/yolov8_accuracy.html)


## Third-Party License

*This project utilizes third-party software and libraries. The licenses for these dependencies are outlined below:*

- **Model**: [Yolov8 Detection Models from Ultralytics](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Code and Pre/Post-Processing**: The DetectionValidator API was sourced from their [GitHub](https://github.com/ultralytics/ultralytics) 🔗
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

## Summary

This guide offers a quick and easy way to validate the accuracy of a pretrained YOLOv8 model on the COCO dataset using MemryX accelerators. You can use the provided Python script to perform the validation. Download the full code and the pre-compiled DFP file to get started immediately.
