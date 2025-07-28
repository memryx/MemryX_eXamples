# Wireframe Detection Using M-LSD (Large) and QT
The Wireframe Inference example demonstrates real-time wireframe detection using an M-LSD (Large) model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

This example also showcases using Qt for the GUI and demonstrates how to integrate MemryX's accelerator (Accl) code within a Qt application, providing a seamless and efficient user interface for wireframe detection.

<p align="center">
  <img src="assets/wireframe.png" alt="Wireframe Detection Example" width="65%" />
</p>

## Overview

| Property             | Details                                                                 
|----------------------|-------------------------------------------------------------------------
| **Model**            | [M-LSD (Large)](https://arxiv.org/abs/2106.00186) 🔗                                            
| **Model Type**       | Line Segment Detection                                                       
| **Framework**        | [TensorFlow Lite](https://www.tensorflow.org/) 🔗                                                   
| **Model Source**     | [M-LSD GitHub Repository](https://github.com/navervision/mlsd) 🔗, [PINTO Model Zoo - M-LSD](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/119_M-LSD) 🔗
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/M_LSD_512_512_4_tflite.zip)                                           
| **Output**           | Wireframe coordinates and scores                
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)                                         

## Requirements

Before running the application, ensure that OpenCV and PySide6 are installed, especially for the Python implementation. You can install them using the following commands:

```bash
pip install opencv-python==4.11.0.86 PySide6
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/M_LSD_512_512_4_tflite.zip
mkdir -p models
unzip M_LSD_512_512_4_tflite.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>

First create the models folder using the steps below.

```bash
mkdir models 
cd models 

```

You can download the pre-trained M-LSD (Large) model from the following sources:

- [M-LSD GitHub Repository](https://github.com/navervision/mlsd) 🔗
- [PINTO Model Zoo - M-LSD](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/119_M-LSD) 🔗

Ensure you download the **M-LSD_512_large_fp32.tflite** model, then run the command below to rename the file. 

```bash
mv M-LSD_512_large_fp32.tflite M_LSD_512_512_4_tflite.tflite
```

You can use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator. If you prefer, you can download the pre-compiled DFP and skip this step.

```bash
mx_nc -m M_LSD_512_512_4_tflite.tflite --autocrop
```

</details>


### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference using the provided Python script. To run the Python example for real-time wireframe detection using MXA, simply execute the following command:

```bash
cd src
python run_wireframe.py
```

You can specify the model path and DFP (Compiled Model) path with the following options:

* `-d` or `--dfp`: Path to the compiled DFP file (default is `../assets/M_LSD_512_512_4_tflite.dfp`)
* `--premodel`: Path to the pre-processing model file
* `--postmodel`: Path to the post-processing model file

For example, to run with specific pre and post models along with a DFP file, use:

```bash
cd scr
python run_wireframe.py --premodel <premodel_path> --postmodel <postmodel_path> -d <dfp_path>
```

If no arguments are provided, the script will use the default paths.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [M-LSD (Large) Model from the M-LSD GitHub Repository](https://github.com/navervision/mlsd) 🔗  
  - License: [Apache-2.0](https://github.com/navervision/mlsd?tab=Apache-2.0-1-ov-file#readme) 🔗

- **Additional Model Source**: [PINTO Model Zoo - M-LSD](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/119_M-LSD) 🔗  
  - License: [Apache-2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/119_M-LSD/LICENSE) 🔗

- **Code Reuse**: Some code components, including pre/post-processing, were sourced from the M-LSD code provided on [PINTO Model Zoo - M-LSD](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/119_M-LSD) 🔗  
  - License: [Apache-2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/119_M-LSD/LICENSE) 🔗

## Summary

This guide offers a quick and easy way to run wireframe detection using an M-LSD (Large) model on MemryX accelerators. You can use the Python implementation to perform real-time inference. Additionally, the example demonstrates how to integrate MemryX's accelerator code within a Qt application for a seamless and efficient user interface. Download the full code and the pre-compiled DFP file to get started immediately.
