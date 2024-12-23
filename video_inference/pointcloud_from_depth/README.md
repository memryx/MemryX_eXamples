# 3D Point Cloud from Depth Estimation

The **Point Cloud from Depth Estimation** example demonstrates real-time depth inference using the pre-trained MiDaS v2 Small model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started with generating point clouds from depth data.

<p align="center">
  <img src="assets/point_cloud.gif" alt="Point-cloud Example" width="55%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [MiDaS v2 Small](https://arxiv.org/pdf/1907.01341) 🔗 
| **Model Type**       | Depth Estimation                                                        |
| **Framework**        | [TensorFlow](https://www.tensorflow.org/) 🔗
| **Model Source**     | [Download from TensorFlow Hub](https://www.kaggle.com/models/intel/midas) 🔗
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/1p1/depth_estimation_using_midas.zip)                                           
| **Input**            | 256x256 (default)                                                       
| **Output**           | Depth map (matches input resolution), Point Cloud Visualization         |
| **Application**      | Real-time point cloud generation and visualization from depth data      |
| **OS**               | Linux, Windows |
| **License**          | [MIT](LICENSE.md)                                         

## Requirements

### Linux

Before running the application, ensure that **OpenCV**, **Open3D**, and **curl** are installed. You can install the necessary libraries using the following commands:

```bash
pip install opencv-python open3d
sudo apt install curl
```

### Windows

On Windows, first make sure you have installed [Python 3.11](https://apps.microsoft.com/detail/9nrwmjp3717k)🔗

Then open the `src/python_windows/` folder and double-click on `setup_env.bat`. The script will install all requirements automatically.


### Linux Python Requirements

Although the MemryX SDK supports Python 3.12, Open3D is currently incompatible with Python versions higher than 3.11. Therefore, to run this example, you'll need Python 3.11 installed. You can check your current Python version with the following command:

```bash
python --version
```

If your version is 3.12 or higher, follow the steps below to install Python 3.11:

#### Installing Python 3.11 on Linux

1. Add the deadsnakes PPA (for Ubuntu-based systems):
    ```bash
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    ```

2. Install Python 3.11:
    ```bash
    sudo apt install python3.11 python3.11-venv python3.11-dev
    ```
3. Verify the installation:
    ```bash
    python3.11 --version
    ```

4. (Optional) Set up a virtual environment to ensure you're using Python 3.11 for this project:
    ```bash
    python3.11 -m venv mxenv
    source mxenv/bin/activate
    ```

**Important Note**: Please ensure that the MemryX SDK is installed in the new Python installation or virtual environment. For instructions, refer to the [SDK tool installation](https://developer.memryx.com/docs/get_started/install_tools.html) page for guidance.


## Running the Application

### Step 1: Download Pre-compiled DFP

#### Windows

[Download](https://developer.memryx.com/example_files/1p1/depth_estimation_using_midas.zip) and place the .dfp file in the `python_windows/models/` folder.

#### Linux

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/1p1/depth_estimation_using_midas.zip
mkdir -p models
unzip depth_estimation_using_midas.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained MiDaS v2 Small model from TensorFlow Hub:

```bash
curl -L -o ./midas_v2_small.tar.gz https://www.kaggle.com/api/v1/models/intel/midas/tfLite/v2-1-small-lite/1/download
tar -xzf ./midas_v2_small.tar.gz -C ./
mkdir -p models
mv 1.tflite ./models/midas_v2_small.tflite
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -m models/midas_v2_small.tflite
```

</details>


### Step 2: Run the Script/Program

To run the Python example for real-time point cloud generation from depth data using MX3, simply execute the following command:

```bash
python src/python/run_pointcloud_from_depth.py
```
You can specify the model path and DFP (Compiled Model) path with the following options:

* `-m` or `--model`: Path to the model file (default is models/midas_v2_small.tflite)
* `-d` or `--dfp`: Path to the compiled DFP file (default is models/midas_v2_small.dfp)

For example, to run with a specific model and DFP file, use:

```bash
python src/python/run_pointcloud_from_depth.py -m <model_path> -d <dfp_path>
```

If no arguments are provided, the script will use the default model and DFP paths.

#### Windows

On Windows, you can instead just **double-click the `run.bat` file** instead of invoking the python interpreter on the command line.


### Interactive Point Cloud Visualization

The point cloud window is fully interactive, allowing you to pan, zoom, and tilt the view using your mouse:

- **Pan**: Left-click and drag to move the point cloud.
- **Zoom**: Scroll up or down with your mouse wheel to zoom in or out.
- **Tilt**: Right-click and drag to change the angle of the view.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [MiDaS v2 Small (TF Lite) from kaggle](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗 
  - License: [MIT](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from the MiDaS v2 Small model provided on [Kaggle](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1)  
  - License: [MIT](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗

## Summary

This guide offers a quick and easy way to generate point clouds from depth estimation using the MiDaS model on MemryX accelerators. You can use the Python implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
