# MxFit - A repcounting web-application

The **MxFit** is a simple web-application based on Yolov8 pose estimation that serves as exercise rep counter. Application showcases the ease of integration with using memryX runtime python API. 

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolov8m-pose](https://docs.ultralytics.com/models/yolov8/)                                            |
| **Model Type**       | Pose Estimation                                                        |
| **Framework**        | [ONNX](https://onnx.ai/)                                                   |
| **Model Source**     | [Download from Ultralytics GitHub or docs](https://docs.ultralytics.com/models/yolov8/) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/1p1/YOLO_v8_medium_pose_640_640_3_onnx.zip)                                          |
| **Model Resolution** | 640x640                                                       |
| **Output**           | Person bounding boxes and pose landmark coordinates |
| **OS**               | Linux |
| **License**          | [AGPL](LICENSE.md)                                       |

## Demo

![Demo GIF](assets/MxFit.gif)

## Requirements

Before running the application, ensure that Python and required packages are installed. You can install the required packages using the following command:

The `mx` env created for sdk installation can be used

```bash
. mx/bin/activate
pip install -r freeze

```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/1p1/YOLO_v8_medium_pose_640_640_3_onnx.zip
mkdir -p models
unzip YOLO_v8_medium_pose_640_640_3_onnx.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained YOLOv8m-pose model and export it to ONNX:

You can use the following code to download the pre-trained yolov8m-pose.pt model and export it to ONNX format:

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-pose.pt")  # load an official model

# Export the model
model.export(format="onnx")
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -v -m yolov8m-pose.onnx --autocrop -c 4
```

Output:
The MemryX compiler will generate two files:

* `yolov8m-pose.dfp`: The DFP file for the main section of the model.
* `yolov8m-pose_post.onnx`: The ONNX file for the cropped post-processing section of the model.

Additional Notes:
* `-v`: Enables verbose output, useful for tracking the compilation process.
* `--autocrop`: This option ensures that any unnecessary parts of the ONNX model (such as pre/post-processing not required by the chip) are cropped out.

</details>

### Step 2: Run the Application

With the compiled model, you can now run the application

Simply execute the following command:

```bash
cd src
python main.py
```
Following output will be printed on the terminal, `ctrl + click` on the link to open the webapp

```
INFO:     Started server process [17126]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

```


## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Yolov8M-pose from Ultralytics GitHub](https://docs.ultralytics.com/models/yolov8/) 🔗 
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Code Reuse - Model Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from their [GitHub](https://github.com/ultralytics/ultralytics)  
  - [AGPLv3](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 🔗

- **Code Reuse - Tracker**: Some code components, including byte track, were sourced from [ByteTrack - github repository](https://github.com/ifzhang/ByteTrack?tab=readme-ov-file) 🔗
  - [MIT](https://github.com/ifzhang/ByteTrack?tab=readme-ov-file) 🔗


