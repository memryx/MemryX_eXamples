# Cartoonizer Using facial-cartoonizer model

The **Cartoonizer** example demonstrates real-time "cartoonization" of a video stream on the MemryX MX3 using an open-source model. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/cartoonizer.png" alt="Cartoonizer Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | FacialCartoonization (https://github.com/SystemErrorWang/FacialCartoonization)                                      |
| **Model Type**       | Cartoonizer                                               |
| **Framework**        | [ONNX](https://onnx.ai/)                                   |
| **Model Source**     | [Download here](https://github.com/SystemErrorWang/FacialCartoonization/blob/master/weight.pth)       |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/Facial_cartoonizer_512_512_3_onnx.zip)         |
| **Model Resolution** | 512x512                                                      |
| **Output**           | cartoonized version of the input image |
| **OS**               | Linux, Windows |
| **License**          | [License](LICENSE.md)                           |

## Requirements

### Linux

Before running the application, ensure that Python, OpenCV, and the required packages are installed. You can install OpenCV and the pyfakewebcam using the following commands:

```bash
pip install opencv-python==4.11.0.86
pip install pyfakewebcam==0.1.0
```

### Windows

On Windows, first make sure you have installed [Python 3.11](https://apps.microsoft.com/detail/9nrwmjp3717k)🔗

Then open the `src/python_windows/` folder and double-click on `setup_env.bat`. The script will install all requirements automatically.

## Running the Application

### Step 1: Download Pre-compiled DFP

#### Windows

[Download](https://developer.memryx.com/example_files/cartoonizer.zip) and place the .dfp file in the `python_windows/models/` folder.


#### Linux

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/Facial_cartoonizer_512_512_3_onnx.zip
mkdir -p models
unzip Facial_cartoonizer_512_512_3_onnx.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pretrained model weights (weight.pth) from the FacialCartoonization GitHub repository

```bash
wget https://github.com/SystemErrorWang/FacialCartoonization/blob/master/weight.pth
```

To export the model to ONNX format:

1. Run the following command:

  ```bash
  python src/utils/generate_onnx.py
  ```

2. This will generate the ONNX model inside the `models/` directory.

3. Change to the models directory:

  ```bash
  cd models
  ```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m facial-cartoonizer_512.onnx --autocrop -c 4 --dfp_fname Facial_cartoonizer_512_512_3_onnx.dfp
```

Output:
The MemryX compiler will generate dfp file:

* `facial-cartoonizer_512.dfp`: The DFP file for the main section of the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.

</details>

### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python.

To run the example for real-time Cartoonizer using MX3, follow these steps:

Simply execute the following command:

```bash
cd src/python/
python run_facial_cartoonizer.py
```
Command-line Options:
You can specify the model path and DFP (Compiled Model) path using the following options:

* `-d` or `--dfp`:  Path to the compiled DFP file (default is models/Facial_cartoonizer_512_512_3_onnx.dfp) or the original model

Example:
To run with a specific model and DFP file, use:

```bash
python run_facial_cartoonizer.py -d <dfp_path> 
```

If no arguments are provided, the script will use the default paths for the model and DFP.

#### Windows

On Windows, you can instead just **double-click the `run.bat` file** instead of invoking the python interpreter on the command line.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [From from GitHub](hhttps://github.com/SystemErrorWang/FacialCartoonization) 🔗 
  - License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) 🔗

- **Preview**: ["Two Baseball Players Talking to Each Other" on Pexels](https://www.pexels.com/video/two-baseball-players-talking-to-each-other-5182642/)  
  - License: [Pexels License](https://www.pexels.com/license/)
  
## Summary

This guide offers a quick and easy way to run Cartoonizer using the facial-cartoonizer model on MemryX accelerators. You can use Python implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
