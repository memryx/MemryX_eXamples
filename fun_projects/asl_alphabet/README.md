# ASL Alphabet Detection with Mediapipe Hand Landmarks

This application uses Mediapipe Palm and Hand models ('full' variants) to detect hands, draw keypoints on them, and identify ASL alphabet gestures in real time based on the detected keypoints. This guide provides setup instructions, model details, and code snippets to help you quickly get started.

<p align="center">
  <img src="assets/asl_demo.gif" width="90%" />
</p>

## Overview

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | [MediaPipe Palm detection model](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#palm-detection-model)🔗, [MediaPipe Hand Landmark model](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#hand-landmark-model)🔗
| **Model Type**       | Palm Detection & Hand Landmark Models
| **Framework**        | TFLite
| **Model Source**     | [Palm Detection (Full)](https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite)🔗⬇️ ,  [Hand Landmark (Full)](https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite)🔗⬇️ from the [google-edge-ai/mediapipe repository](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md#hands)🔗
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/1p1/mediapipe_hands.zip)
| **Input**            | Input size for Palm Detection model: (192,192,3), Input size for Hand Landmark model: (224,224,3)
| **Output**           | Output from HandLandmark model: bounding boxes, landmarks, rotated landmarks, handedness, confidence 
| **License**          | [MIT License](LICENSE.md)


## Requirements

### Linux

Before running the application, ensure that **OpenCV** is installed

You can install OpenCV using the following command:

```bash
pip install opencv-python
```


## Running the Application (Linux)

### Step 1: Download or Compile DFP

#### Linux

To download and unzip the precompiled DFPs, use the following commands:

```bash
cd models
wget https://developer.memryx.com/example_files/1p1/mediapipe_hands.zip
unzip mediapipe_hands.zip
```

<details>
<summary> (Optional) Download and Compile the Model Yourself </summary>

If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained 

* Palm Detection and HandLandmark models from from the [google-edge-ai/mediapipe repository](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md#hands)🔗

```bash
wget https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite
wget https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -m hand_landmark_full.tflite palm_detection_full.tflite --autocrop
```

**NOTE:** if you compile the DFP yourself, the NeuralCompiler will create a cropped post-processing model. This model is just simple data organziation operations, so our `MxHandPose.py` actually forgoes use of the `post.tflite` and uses plain numpy functions. Thus it is safe to delete the post model file.

</details>

---

Your folder structure should now be:
```
|- README.md
|- LICENSE.md
|- data/
|  |- gesture_clf.pkl
|
|- models/
|  |- models.dfp
|
|- src/
|  |- python/
|      |- mp_handpose.py
|      |- mp_palmdet.py
|      |- MxHandPose.py
|      |- run.py
|      |- extra.py
```



### Step 2: Run the Program

#### Linux

To run on Linux, make sure your python env is activated and simply execute the following commands:

```bash
cd src/python/
python run.py
```
A window will appear to display the result of real time inference on the MX3. The window will show a real time prediction of an ASL sign in green whenever a hand is detected in the frame. For your reference, the below chart can be used to lean the ASL alphabet.

<p align="center">
  <img src="assets/ASL_Alphabet_Chart.png" width="50%" />
</p>

To spell a word or phrase, hold an ASL alphabet sign until you see the corresponding letter appear at the bottom of the window. Then, move on to the next letter!

Hit 'c' to clear the word or phrase at the bottom of the window.

Hit 'Backspace' to delete the last letter in your word or phrase.

Hit the Spacebar to separate words.

Hit 'q' to quit the program!


## Third-Party Licenses

*This project utilizes third-party software and libraries. The licenses for these dependencies are outlined below:*

- **Models**: [MediaPipe Palm detection model and MediaPipe Hand Landmark model from google-ai-edge/mediapipe repository](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md#hands)🔗
    - License : [Apache 2.0 License](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) 🔗
- **Code Reuse**: Preprocessing and postprocessing code was used from the [opencv repository](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe)🔗
    - License : [Apache 2.0 License](https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/LICENSE)🔗
- **Code Reuse**: ASL gesture data and gesture classifier code was used from the [Gesture-Recognition repository](https://github.com/aqua1907/Gesture-Recognition/tree/master)🔗
