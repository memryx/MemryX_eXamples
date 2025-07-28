# Virtual Painter Using Mediapipe's Palm Detection & Hand Landmark Models

The **Virtual Painter** enables the user to virtually paint in the air in real-time using the Palm Detection & Hand Landmark models on the MemryX accelerator. This guide provides setup instructions, model details, and code snippets to help you quickly get started.

## Overview

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | [MediaPipe Palm detection model](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#palm-detection-model)🔗, [MediaPipe Hand Landmark model](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#hand-landmark-model)🔗
| **Model Type**       | Palm Detection & Hand Landmark Models
| **Framework**        | TFLite
| **Model Source**     | [Palm Detection Model Lite](https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite)🔗⬇️ ,  [Hand Landmark Model Lite](https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite)🔗⬇️ from the [google-edge-ai/mediapipe repository](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md#hands)🔗
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/2p0/virtual_painter_using_palmdet_handlandmark.zip)
| **Input**            | Input size for Palm Detection Model: (192,192,3), Input size for Hand Landmark model : (224,224,3)
| **Output**           | Output from HandLandmark model: bounding boxes, landmarks, rotated landmarks, handedness, confidence 
| **License**          | [MIT License](LICENSE.md)


## Demo

<p align="center">
  <img src="assets/virtual_painter.gif" alt="Virtual Painter Example" width="35%" />
</p>



## Requirements

Before running the application, ensure that **OpenCV** is installed

You can install OpenCV using the following command:

```bash
pip install opencv-python==4.11.0.86
```

## Running the Application (Linux)

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:

```bash
cd assets
wget https://developer.memryx.com/example_files/2p0/virtual_painter_using_palmdet_handlandmark.zip
unzip virtual_painter_using_palmdet_handlandmark.zip
```

<details>
<summary> (Optional) Download and Compile the Model Yourself </summary>

If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained 

* Palm Detection and HandLandmark models from from the [google-edge-ai/mediapipe repository](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md#hands)🔗

```bash
wget https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite -O palm_detection_lite.tflite
wget https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite  -O hand_landmark_lite.tflite
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -m hand_landmark_lite.tflite palm_detection_lite.tflite --autocrop -c 4
```
</details>

Your folder structure should now be:
```
|- README.md
|- LICENSE.md
|- assets/
|  |- gestures_data.pkl
|  |- virtual_painter.gif
|  |- model_1_palm_detection_lite_post.tflite    
|  |- models.dfp
|  |- settings.json
|
|- src/
|  |- python 
|      |- mp_handpose.py
|      |- mp_palmdet.py
|      |- MxHandPose.py
|      |- virtual_painter.py
```

The assets folder in the above structure contain only the **essential** files required for the execution of the script. If you have decided to download and compile the model yourself, there may be additional files present in the assets folder.


### Step 2: Run the Script/Program

With the compiled model, you can begin drawing with your hand! 

#### Python

To run the Python example and draw in real-time using MX3, simply execute the following commands:

```bash
cd src/python/
python virtual_painter.py
```

Hit 'c' from the keyboard to clear the screen and 'q' to quit the program!

**Additional settings**
By changing the `assets/settings.json` file the `command_hand` and the `brush_hand` can be changed.
By default `command_hand` is the the left hand and `brush_hand` is the right hand.

**To Draw**
When a fist is formed with the `command_hand`, then the user is in `Draw` mode, you can draw with the index finger of the `brush_hand`.

**To select color or change the brush size**
When the `command_hand`'s palm is detected on the screen, drawing is paused. This is the `StandBy` mode. You can use your `bursh_hand`'s index finger to change your brush size and pick a colour in this mode.

Helpful hint: Look at the the example provided in the Demo and create your own art!

## Third-Party Licenses

*This project utilizes third-party software and libraries. The licenses for these dependencies are outlined below:*

- **Models**: [MediaPipe Palm detection model and MediaPipe Hand Landmark model from google-ai-edge/mediapipe repository](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md#hands)🔗
    - License : [Apache 2.0 License](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) 🔗
- **Code Reuse**: The foundation of this example was borrowed from [AI-Virtual-Painter repository](https://github.com/darthdaenerys/AI-Virtual-Painter) 🔗 
    - License : [MIT License](https://github.com/darthdaenerys/AI-Virtual-Painter/blob/master/LICENSE) 🔗
- **Code Reuse**: Preprocessing and postprocessing code was used from the [opencv repository](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe)🔗
    - License : [Apache 2.0 License](https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/LICENSE)🔗


## Summary

This guide provides a quick and easy way to Virtually Draw using the MediaPipe Palm detection model and Hand Landmark models on MemryX accelerators. With the Python implementation, you can paint in real-time. Simply download the full code and the pre-compiled DFP file to get started immediately.
