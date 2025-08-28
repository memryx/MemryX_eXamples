<!-- Announcement Banner -->
> **📢 Announcement:** **MemryX SDK 2.0** is now released!  
> Please update to the [latest SDK version](https://developer.memryx.com/get_started/index.html) **before** proceeding with any of the examples.
<!-- End Banner -->

<!-- Define the reusable badges -->
[python-badge]: https://img.shields.io/badge/Python-green "Python"
[cpp-badge]: https://img.shields.io/badge/C++-blue "C++"

<picture>
  <source srcset="figures/mx_examples.png" media="(prefers-color-scheme: dark)">
  <source srcset="figures/mx_examples_light.png" media="(prefers-color-scheme: light)">
  <img src="figures/mx_examples_light.png" alt="MemryX eXamples">
</picture>


[![MemryX SDK](https://img.shields.io/badge/MemryX%20SDK-2.0-brightgreen)](https://developer.memryx.com)
[![Python Versions](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![C++](https://img.shields.io/badge/C++-17-blue)](https://en.cppreference.com)
[![ONNX](https://img.shields.io/badge/ONNX-gray)](https://onnx.ai)
[![Keras](https://img.shields.io/badge/Keras-gray)](https://keras.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-gray)](https://www.tensorflow.org)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlowLite-gray)](https://www.tensorflow.org/lite)



# MemryX eXamples

Welcome to **MemryX eXamples**, a collection of end-to-end AI applications and tasks powered by MemryX hardware and software solutions. Whether you're performing real-time video inference, exploring fun AI projects, or generating text, these examples provide practical, hands-on use cases to help you fully leverage MemryX technology. For detailed guides and tutorials, visit the [MemryX Developer Hub](https://developer.memryx.com/index.html).

## Before You Start

To ensure a smooth experience with MemryX solutions, follow these steps before diving into the examples:

- **Explore the [Developer Hub](https://developer.memryx.com/index.html):** Your gateway to comprehensive documentation for MemryX hardware and software.
- **Install the [MemryX SDK](https://developer.memryx.com/get_started/install.html):** Set up the essential tools and drivers to begin using MemryX accelerators.
- **Check out our [Tutorials](https://developer.memryx.com/tutorials/tutorials.html):** Step-by-step instructions for various use cases and end-to-end applications.
- **Explore the [Model Explorer](https://developer.memryx.com/model_explorer/models.html):** A great starting point for discovering models compiled and tested on MemryX accelerators.

## Get Started

### Step 1: Prepare Your System and Install the MemryX SDK

Before working with the examples, ensure your system is correctly set up by installing the MemryX SDK.
Follow the detailed instructions here: [**MemryX SDK Get Started Guide**](https://developer.memryx.com/get_started/index.html).

### Step 2: Clone the MemryX eXamples Repository

Clone this repository plus any linked submodules with:

```bash
git clone --recursive https://github.com/memryx/memryx_examples.git
```

## Example Categories

**Note:** Applications marked with **📝** have tutorials available. Clicking on the icon will take you directly to the tutorial page.

### Real-Time Video Inference 🎥
Leverage MemryX accelerators for **real-time video processing** tasks. These applications demonstrate how to run models efficiently on live video streams.

| Application                                         | Description                              | Models        | Code                   | OS  | Preview |
|-----------------------------------------------------|------------------------------------------|---------------|------------------------|-----|-----------|
| [**Depth Estimation using MiDaS**](video_inference/depth_midas/README.md) [📝](https://developer.memryx.com/tutorials/realtime_inf/realtime_depth.html) | Estimate depth from a video stream       | MiDaS         | ![python-badge] ![cpp-badge] |  <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" alt="Windows" width="20" height="20"> | <img src="video_inference/depth_midas/assets/depth.png" alt="Depth Preview" height="50"> |
| [**Object Detection using YOLOv7**](video_inference/singlestream_objectdetection_yolov7Tiny/README.md) [📝](https://developer.memryx.com/tutorials/realtime_inf/realtime_od.html) | Detect objects in real time              | YOLOv7 (Tiny) | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/singlestream_objectdetection_yolov7Tiny/assets/objectDetection_yolov7tiny.png" alt="Yolov7 Tiny Object Detection Preview" height="50"> |
| [**Object Detection using CenterNet**](video_inference/centernet/README.md) [📝](https://developer.memryx.com/tutorials/realtime_inf/autocrop_inf/autocrop_centernet.html) | Detect objects in real time              | CenterNet     | ![cpp-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/centernet/assets/centernet.gif" alt="CenterNet Preview" height="50"> |
| [**Object Detection using YoloX**](video_inference/object_detection_yolox/README.md) | Detect objects in real time              | YoloX (Medium)| ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/object_detection_yolox/assets/yolox.gif" alt="YoloX Object Detection Preview" height="50"> |
| [**Vehicle Detection**](video_inference/vehicle_detection/README.md) | Detect vehicle in real time | Vehicle-Detection-0200 | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/vehicle_detection/assets/output.gif" alt="vehicle Detection Preview" height="50"> |
| [**Segmentation using YOLOv8**](video_inference/segmentation_yolov8/README.md) | Perform instant segmentation on video in real time | YOLOv8 Nano Segmentation | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/segmentation_yolov8/assets/segmentation.png" alt="Yolov8n Segmentation Preview" height="50"> |
| [**Pose Estimation using YOLOv8**](video_inference/pose_estimation_yolov8/README.md) [📝](https://developer.memryx.com/tutorials/realtime_inf/realtime_pose.html) | Estimate human pose from video           | YOLOv8 (Medium)| ![python-badge] ![cpp-badge]  |<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" alt="Windows" width="20" height="20">| <img src="video_inference/pose_estimation_yolov8/assets/pose_estimation.gif" alt="Yolov8 Pose Estimation Preview" height="50"> |
| [**3D Point Cloud from Depth Estimation**](video_inference/pointcloud_from_depth/README.md) | Generate real-time point clouds from depth data | MiDaS | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" alt="Windows" width="20" height="20"> | <img src="video_inference/pointcloud_from_depth/assets/point_cloud.gif" alt="Point-cloud Preview" height="50"> |
| [**Wireframe detection Using M-LSD and QT**](video_inference/wireframe/README.md) | Perform Line segment detection in real time | M-LSD (Large) | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/wireframe/assets/wireframe.png" alt="Wireframe Preview" height="50"> |
| [**Face Detection & Emotion Classification**](video_inference/face_emotion_detection/README.md) [📝](https://developer.memryx.com/tutorials/realtime_inf/realtime_multimodel.html) | Detect faces and classify emotions       | Multiple Models | ![python-badge] ![cpp-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> |  <img src="video_inference/face_emotion_detection/assets/face_emotion.png" alt="Face Detection & Emotion Classification Preview" height="50"> |
| [**Person Tracking using YOLOv7**](video_inference/singlestream_peopletracking_yolov7Tiny/README.md) | Track unique people across video frames | YOLOv7 (Tiny) | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/singlestream_peopletracking_yolov7Tiny/assets/people_counting.gif" alt="Person Tracking Preview" height="50"> |
| [**Face Landmarks Detection**](video_inference/realtime_facelandmark_detection/README.md) | Detect presence of a face and its landmarks | BlazeFace and FaceMesh | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/realtime_facelandmark_detection/assets/sample.gif" alt="Landmarks Preview" height="50"> |
| [**Mediapipe Hand Landmarks**](video_inference/mediapipe_hands/README.md) | Detect hands and draw a "skeleton" of landmarks | PalmDet and HandPose | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" alt="Windows" width="20" height="20"> | <img src="video_inference/mediapipe_hands/assets/hand.gif" alt="Hand Preview" height="50"> |
| [**Interactive Realtime Multi-Face Recognition**](video_inference/realtime_multiface_recognition/README.md) | Interactive app for face recognition | Multiple Models | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/realtime_multiface_recognition/assets/demo.gif" alt="Face Recognition App" height="50"> |
| [**Intrusion Detection**](video_inference/intrusion_detection/README.md) | Detect any intruding object in a ROI | Yolov8 and ByteTrack| ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="video_inference/intrusion_detection/assets/intrusion.gif" alt="Intrusion Preview" height="50"> |


### Image Inference 🖼️
Explore models performing inference on static images and data. These examples demonstrate how to leverage the MXA to process large amounts of data.

| Application                                         | Description                              | Models        | Code                   | OS  | Preview |
|-----------------------------------------------------|------------------------------------------|---------------|------------------------|-----|-----------|
| [**Satellite Object Detection with Oriented Boxes**](image_inference/oriented_bounding_boxes/README.md)       | Detect oriented bounding boxes on satellite images | YoloV8m-OBB | ![python-badge]  | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="image_inference/oriented_bounding_boxes/assets/parking_lot_output.png" alt="OBB Preview" height="50"> |
| [**Face Detection + Recognition**](image_inference/face_recognition/README.md)       | Perform face detection + recognition     | YoloV8n-Face + FaceNet | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="image_inference/face_recognition/assets/face.png" alt="Face Preview" height="50"> |


### Open-Vocabulary 📖
Explore how the MemryX accelerators can be used for open-vocabulary tasks. The examples below demonstrate how to run models using MemryX hardware.  

| Application                                         | Description                              | Models        | Code                   | OS  | Preview |
|-----------------------------------------------------|------------------------------------------|---------------|------------------------|-----|-----------|
| [**Open-Vocab Seg with YoloE**](open_vocabulary/yoloe/README.md) | open-vocabulary, zero-shot object detection and segmentation      | yoloe-v8s-seg| ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="open_vocabulary/yoloe/assets/yoloe.gif" alt="YoloE App Preview" height="50"> |


### Multi-Stream Video Inference 🖥️
Maximize performance by running multiple video streams concurrently on MemryX accelerators.

| Application                                         | Description                              | Models        | Code                   | OS  | Preview |
|-----------------------------------------------------|------------------------------------------|---------------|------------------------|-----|-----------|
| [**Multi-Stream Object Detection using YOLOv8S**](multistream_video_inference/object_detection_yolov8/README.md) [📝](https://developer.memryx.com/tutorials/multistream_realtime_inf/multistream_od_yolov8s.html) | Detect objects across multiple streams      | YOLOv8 (Small) | ![python-badge] ![cpp-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="multistream_video_inference/object_detection_yolov8/assets/yolov8_objectDetection.png" alt="Yolov8 Object Detection Preview" height="50"> |
| [**Multi-Stream Object Detection using YOLOv7Tiny**](multistream_video_inference/multistream_objectdetection_yolov7Tiny/README.md) [📝](https://developer.memryx.com/tutorials/multistream_realtime_inf/multistream_od.html) | Detect objects across multiple streams      | YOLOv7 (Tiny)  | ![python-badge] ![cpp-badge]  | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="multistream_video_inference/multistream_objectdetection_yolov7Tiny/assets/yolov7_objectDetection_multistream.png" alt="Yolov7-Tiny Object Detection Preview" height="50"> |


### Multi-DFP Applications 🖥️
Run multiple models simultaneously on separate DFPs within a single application.

| Application                                         | Description                              | Models        | Code                   | OS  | Preview |
|-----------------------------------------------------|------------------------------------------|---------------|------------------------|-----|-----------|
| [**Cartoonizer and Pose (Side-by-Side)**](multi_dfp_application/cartoonizer_pose/README.md) | Real-time cartoon effects and pose detection simultaneously    | Facial-Cartoonizer & YOLOv8s-pose | ![python-badge] ![cpp-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="multi_dfp_application/cartoonizer_pose/assets/cartoon_pose.png" alt="Cartoon & PoseEstimation Preview" height="50"> |
| [**Cartoonizer and Pose (Selectable Overlays)**](multi_dfp_application/cartoonizer_pose_overlay/README.md) | Real-time overlay of cartoon and pose estimation | Facial-Cartoonizer & YOLOv8s-pose | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="multi_dfp_application/cartoonizer_pose_overlay/assets/overlay.png" alt="Overlay Cartoon & PoseEstimation Preview" height="50"> |

### Optimized Multi-Stream Apps 👀
Configurable many-stream video applications with USB, RTSP, and video file support, using optimized C++ pre/post-processing.

| Application                                         | Description                              | Models        | Code                   | OS  | Preview |
|-----------------------------------------------------|------------------------------------------|---------------|------------------------|-----|-----------|
| [**Object Detection with YOLOv8**](optimized_multistream_apps/yolov8_object_detection/README.md) | High performance, configurable C++ object detection      | YOLOv8 (Nano) | ![cpp-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="optimized_multistream_apps/yolov8_object_detection/assets/result.png" alt="Yolov8 Object Detection App Preview" height="50"> |
| [**PCB Defect Detection**](optimized_multistream_apps/yolov8_pcb_defect_detection/README.md) | Detect defects on printed circuit boards in real time across multiple video streams | Retrained YOLOv8 (Small) | ![cpp-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="optimized_multistream_apps/yolov8_pcb_defect_detection/assets/pcb_defect.png" alt="PCB Defect Detection App Preview" height="50"> |



### Fun Projects 🤖
Explore interactive and engaging AI-powered applications in our **fun projects** section.

| Application                                         | Description                              | Models        | Code                   | OS  | Preview |
|-----------------------------------------------------|------------------------------------------|---------------|------------------------|-----|-----------|
| [**Chrome Dino Game**](fun_projects/chrome_dino_game/README.md) [📝](https://developer.memryx.com/tutorials/fun_projects/dino_game.html) | Control the Chrome Dino Game using palm detection | Palm Detection  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="fun_projects/chrome_dino_game/assets/dino_game.gif" alt="Dino Game Preview" height="50"> |
| [**Deep Reinforcement Learning with Mario**](fun_projects/mario_rl/README.md) | Play Mario with a Reinforcement Learning Agent            | Custom    | ![python-badge]  | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="fun_projects/mario_rl/assets/mario.gif" alt="Mario Game Preview" height="50"> |
| [**Aimbot**](fun_projects/aimbot/README.md) | Automatic aim and click for Windows games            | YOLOv7 (Tiny)    | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" alt="Windows" width="20" height="20"> | <img src="fun_projects/aimbot/assets/aimbot_demo.gif" alt="AimBot Preview" height="50"> 
| [**Repcounting Web-application**](fun_projects/MxFit/README.md) | Workout repcounting web-application            | YOLOv8 Pose (medium)    | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="fun_projects/MxFit/assets/MxFit.gif" alt="MxFit Preview" height="50"> 
| [**Facial Cartoonizer**](fun_projects/cartoonizer/README.md) | Instantly cartoonize videos in real time | Facial-Cartoonizer | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" alt="Windows" width="20" height="20"> | <img src="fun_projects/cartoonizer/assets/cartoonizer.png" alt="Dino Game Preview" height="50"> |
| [**Virtual Painter**](fun_projects/virtual_painter/README.md) | Virtually Paint using Hand Landmarks in real-time           | Palm Detection & Hand Landmark    | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="fun_projects/virtual_painter/assets/virtual_painter.gif" alt="Virtual Painter Preview" height="50"> 
| [**ASL Alphabet to Text**](fun_projects/asl_alphabet/README.md) | Use the American Sign Language alphabet to spell words           | Palm Detection & Hand Landmark    | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | <img src="fun_projects/asl_alphabet/assets/asl_demo.gif" alt="ASL Spelling Preview" height="50"> 

### Accuracy Calculation ✅
Measure and evaluate the accuracy of various models using MemryX hardware.

<div style="width: 100%;">

| Task                           | Description                                       | Models           | Code                   | OS |
|--------------------------------|---------------------------------------------------|------------------|------------------------|----|
| [**Classification Accuracy**](accuracy_calculation/classification_resnet50/README.md) [📝](https://developer.memryx.com/accuracy/mlperf_accuracy/resnet50v1.5_mlperf_accuracy.html) | Calculate accuracy for classification models      | ResNet50         | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20">
| [**Object Detection Accuracy**](accuracy_calculation/detect_yolov8/README.md) [📝](https://developer.memryx.com/tutorials/accuracy/yolov8_accuracy/yolov8_accuracy.html) | Calculate accuracy for object detection models    | YOLOv8 (Medium)  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20">
| [**Keras Classifiers Accuracy**](accuracy_calculation/keras_accuracy/README.md) [📝](https://developer.memryx.com/tutorials/accuracy/keras_classifiers_accuracy/keras_accuracy_rst.html) | Calculate Keras classifiers accuracy on the MXA    | Keras applications  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20">

### Audio Inference 🔊

Explore how the MemryX accelerators can be used for audio/speech processing tasks. The examples below demonstrate how to run models using MemryX hardware.

<div style="width: 100%;">
  
| Task                           | Description                                | Models  | Code             | OS  |
|--------------------------------|--------------------------------------------|---------|------------------|-----|
| [**Audio Denoising using UNet**](audio_inference/audio_denoising_cmd/README.md) | Remove noise from speech audio using UNet | UNet    | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> |
| [**Audio Classification using YAMNet**](audio_inference/audio_classification_cmd/README.md) | Identify audio categories using YAMNet | YAMNet  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> |
| [**Audio Classification Web App**](audio_inference/audio_classification_web/README.md) | Classify audio using YAMNet in a web app | YAMNet  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | 
| [**Speech Emotion Recognition Web App**](audio_inference/speech_emotion_recognition/README.md) | Detect emotion from speech (web app) | Light-SERNet  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Linux" width="20" height="20"> | 

## Useful Links

- [Developer Hub](https://developer.memryx.com) — Comprehensive documentation for MemryX hardware and software.
- [DevHub Get Started](https://developer.memryx.com/get_started/index.html) — Guide to set up MemryX software and hardware.
- [Tutorials](https://developer.memryx.com/tutorials/tutorials.html) — Step-by-step instructions for various use cases and applications.
- [FAQ](https://developer.memryx.com/support/faq.html) — Frequently asked questions.
- [Troubleshooting Guide](https://developer.memryx.com/support/troubleshooting/index.html) — Solutions to common issues.

## Contribution Guidelines

We welcome contributions! If you'd like to contribute to this repository or examples, please refer to our [contribution guidelines](guidelines/CONTRIBUTING.md). Feel free to submit pull requests, suggest improvements, or ask questions in the issues section.

## Frequently Asked Questions (FAQ)

**1. How do I install the MemryX SDK?**

Refer to the [SDK Installation Guide](https://developer.memryx.com/get_started/install.html) for a detailed step-by-step guide on setting up the MemryX SDK.

**2. What do I do if an example isn't working?**

Make sure you’ve followed all setup steps. You can also check the [Troubleshooting Guide](https://developer.memryx.com/support/troubleshooting/index.html) for more help, or open an issue in the repository.

**3. Can I contribute to this repository?**

Yes! We welcome contributions. Please refer to our [contribution guidelines](guidelines/CONTRIBUTING.md) for more information on how to contribute.

Happy coding! 😊\
The MemryX Team
