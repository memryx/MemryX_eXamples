<!-- Define the reusable badges -->
[python-badge]: https://img.shields.io/badge/Python-green "Python"
[cpp-badge]: https://img.shields.io/badge/C++-blue "C++"

<picture>
  <source srcset="assets/mx_examples.png" media="(prefers-color-scheme: dark)">
  <source srcset="assets/mx_examples_light.png" media="(prefers-color-scheme: light)">
  <img src="assets/mx_examples_light.png" alt="MemryX eXamples">
</picture>


[![MemryX SDK](https://img.shields.io/badge/MemryX%20SDK-2.1-brightgreen)](https://developer.memryx.com)
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

> [!IMPORTANT]
> **MemryX SDK 2.1** is now released!
> Please update to the [latest SDK version](https://developer.memryx.com/get_started/index.html) **before** proceeding with any of the examples.


### Step 1: Prepare Your System and Install the MemryX SDK

Before working with the examples, ensure your system is correctly set up by installing the MemryX SDK.
Follow the detailed instructions here: [**MemryX SDK Get Started Guide**](https://developer.memryx.com/get_started/index.html).

### Step 2: Clone the MemryX eXamples Repository

Clone this repository plus any linked submodules with:

```bash
git clone --recursive https://github.com/memryx/memryx_examples.git
```

## Example Categories

> [!NOTE]
> Applications marked with **📝** have tutorials available. Clicking on the icon will take you directly to the tutorial page.

<details>
<summary><b>Jump to a category</b></summary>

- [Real-Time Video Inference 🎥](#real-time-video-inference)
- [Open-Vocabulary 📖](#open-vocabulary)
- [Image Inference 🖼️](#image-inference)
- [Multi-Stream Applications 🖥️](#multi-stream-applications)
- [Multi-DFP Applications 🔀](#multi-dfp-applications)
- [Fun Projects 🤖](#fun-projects)
- [Audio Inference 🔊](#audio-inference)
- [Accuracy Calculation ✅](#accuracy-calculation)

</details>

<a id="real-time-video-inference"></a>
### Real-Time Video Inference 🎥
Leverage MemryX accelerators for **real-time video processing** tasks. These applications demonstrate how to run models efficiently on live video streams.


<table>
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="video_inference/singlestream_objectdetection_yolov7Tiny/README.md"><b>YOLOv7</b></a>
      <a href="https://developer.memryx.com/tutorials/realtime_inf/realtime_od.html">📝</a><br/>
      <a href="video_inference/singlestream_objectdetection_yolov7Tiny/README.md">
        <img src="video_inference/singlestream_objectdetection_yolov7Tiny/assets/objectDetection_yolov7tiny.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>COCO object detection</sub><br/>
      <sub>Model: YOLOv7 (Tiny)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/centernet/README.md"><b>CenterNet</b></a>
      <a href="https://developer.memryx.com/tutorials/realtime_inf/autocrop_inf/autocrop_centernet.html">📝</a><br/>
      <a href="video_inference/centernet/README.md">
        <img src="video_inference/centernet/assets/centernet.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>COCO object detection</sub><br/>
      <sub>Model: CenterNet</sub><br/>
      <img alt="C++"   src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/object_detection_yolox/README.md"><b>YoloX</b></a><br/>
      <a href="video_inference/object_detection_yolox/README.md">
        <img src="video_inference/object_detection_yolox/assets/yolox.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>COCO object detection</sub><br/>
      <sub>Model: YoloX (Medium)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/vehicle_detection/README.md"><b>Vehicle Detection</b></a><br/>
      <a href="video_inference/vehicle_detection/README.md">
        <img src="video_inference/vehicle_detection/assets/output.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Vehicle detection</sub><br/>
      <sub>Model: Vehicle-Detection-0200</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="video_inference/segmentation_yolov8/README.md"><b>YOLOv8 Segmentation</b></a><br/>
      <a href="video_inference/segmentation_yolov8/README.md">
        <img src="video_inference/segmentation_yolov8/assets/segmentation.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Instance segmentation</sub><br/>
      <sub>Model: YOLOv8 Nano Segmentation</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/pose_estimation_yolov8/README.md"><b>YOLOv8 Pose</b></a>
      <a href="https://developer.memryx.com/tutorials/realtime_inf/realtime_pose.html">📝</a><br/>
      <a href="video_inference/pose_estimation_yolov8/README.md">
        <img src="video_inference/pose_estimation_yolov8/assets/pose_estimation.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Human pose estimation</sub><br/>
      <sub>Model: YOLOv8 (Medium)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/depth_midas/README.md"><b>MiDaS Depth</b></a>
      <a href="https://developer.memryx.com/tutorials/realtime_inf/realtime_depth.html">📝</a><br/>
      <a href="video_inference/depth_midas/README.md">
        <img src="video_inference/depth_midas/assets/depth.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Monocular depth estimation</sub><br/>
      <sub>Model: MiDaS</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/pointcloud_from_depth/README.md"><b>Depth to 3D</b></a><br/>
      <a href="video_inference/pointcloud_from_depth/README.md">
        <img src="video_inference/pointcloud_from_depth/assets/point_cloud.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Point cloud from depth</sub><br/>
      <sub>Model: MiDaS</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="video_inference/face_emotion_detection/README.md"><b>Face + Emotion</b></a>
      <a href="https://developer.memryx.com/tutorials/realtime_inf/realtime_multimodel.html">📝</a><br/>
      <a href="video_inference/face_emotion_detection/README.md">
        <img src="video_inference/face_emotion_detection/assets/face_emotion.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Face + emotion classification</sub><br/>
      <sub>Multiple Models</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/realtime_facelandmark_detection/README.md"><b>Face Landmarks</b></a><br/>
      <a href="video_inference/realtime_facelandmark_detection/README.md">
        <img src="video_inference/realtime_facelandmark_detection/assets/sample.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Facial landmark tracking</sub><br/>
      <sub>Model: BlazeFace & FaceMesh</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/mediapipe_hands/README.md"><b>Mediapipe Hands</b></a><br/>
      <a href="video_inference/mediapipe_hands/README.md">
        <img src="video_inference/mediapipe_hands/assets/hand.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Hand landmark tracking</sub><br/>
      <sub>Models: PalmDet &amp; HandPose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/wireframe/README.md"><b>Wireframe</b></a><br/>
      <a href="video_inference/wireframe/README.md">
        <img src="video_inference/wireframe/assets/wireframe.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Line / wireframe detection</sub><br/>
      <sub>Model: M-LSD (Large)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="video_inference/traffic_analysis/README.md"><b>Traffic Analysis</b></a><br/>
      <a href="video_inference/traffic_analysis/README.md">
        <img src="video_inference/traffic_analysis/assets/example.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Oriented bounding boxes detection</sub><br/>
      <sub>Model: YOLOv8s-OBB</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/football_cv/README.md"><b>FootballCV</b></a><br/>
      <a href="video_inference/football_cv/README.md">
        <img src="video_inference/football_cv/assets/football_cv.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Football Video Analysis</sub><br/>
      <sub>Models: Yolov8 (small)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/luggage_counting_yolov8/README.md"><b>Luggage counting</b></a><br/>
      <a href="video_inference/luggage_counting_yolov8/README.md">
        <img src="video_inference/luggage_counting_yolov8/assets/luggage_counting.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>COCO object detection</sub><br/>
      <sub>Model: YOLOv8 (Nano)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/ppe_detection_tracking_yolov8/README.md"><b>PPE Detection</b></a><br/>
      <a href="video_inference/ppe_detection_tracking_yolov8/README.md">
        <img src="video_inference/ppe_detection_tracking_yolov8/assets/ppe_detection.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Personal protective equipment detection</sub><br/>
      <sub>Model: YOLOv8 (Nano)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="video_inference/object_blurring_yolov8/README.md"><b>Object Blurring</b></a><br/>
      <a href="video_inference/object_blurring_yolov8/README.md">
        <img src="video_inference/object_blurring_yolov8/assets/object_blurring.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Blur detected persons with YOLOv8</sub><br/>
      <sub>Model: YOLOv8 (Small)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/object_tracking_yolov8/README.md"><b>Object Tracking</b></a><br/>
      <a href="video_inference/object_tracking_yolov8/README.md">
        <img src="video_inference/object_tracking_yolov8/assets/object_tracking.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Number and track objects with YOLOv8 and SuperVision tracking</sub><br/>
      <sub>Model: YOLOv8 (Small)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/realtime_multiface_recognition/README.md"><b>Multi-Face ID</b></a><br/>
      <a href="video_inference/realtime_multiface_recognition/README.md">
        <img src="video_inference/realtime_multiface_recognition/assets/demo.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Interactive multi-face recognition</sub><br/>
      <sub>Multiple Models</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/singlestream_peopletracking_yolov7Tiny/README.md"><b>Person Tracking</b></a><br/>
      <a href="video_inference/singlestream_peopletracking_yolov7Tiny/README.md">
        <img src="video_inference/singlestream_peopletracking_yolov7Tiny/assets/people_counting.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Person tracking + IDs</sub><br/>
      <sub>Model: YOLOv7 (Tiny)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="video_inference/intrusion_detection/README.md"><b>Intrusion Detection</b></a><br/>
      <a href="video_inference/intrusion_detection/README.md">
        <img src="video_inference/intrusion_detection/assets/intrusion.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>ROI intrusion alerts</sub><br/>
      <sub>Models: Yolov8 &amp; ByteTrack</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/rtmpose_estimate/README.md"><b>Rtmpose Estimate</b></a><br/>
      <a href="video_inference/rtmpose_estimate/README.md">
        <img src="video_inference/rtmpose_estimate/assets/out.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Pose Estimate with tracking</sub><br/>
      <sub>Models: Yolox &amp; Rtmpose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="video_inference/fire_detection_yolov8/README.md"><b>Fire Detection</b></a><br/>
      <a href="video_inference/fire_detection_yolov8/README.md">
        <img src="video_inference/fire_detection_yolov8/assets/out.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Fire detection with tracking</sub><br/>
      <sub>Models: Yolov8</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">&nbsp;</td>
  </tr>
</table>


<a id="open-vocabulary"></a>
### Open-Vocabulary 📖
Explore how the MemryX accelerators can be used for open-vocabulary tasks. The examples below demonstrate how to run models using MemryX hardware.

<table>
  <tr>
    <td align="center" valign="top" width="45%">
      <a href="open_vocabulary/yoloe/README.md"><b>YoloE Open-Vocab Seg</b></a><br/>
      <a href="open_vocabulary/yoloe/README.md">
        <img src="open_vocabulary/yoloe/assets/yoloe.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Open-vocabulary detection + segmentation</sub><br/>
      <sub>Model: yoloe-v8s-seg</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="35%">
      <a href="open_vocabulary/clipResNet50/README.md"><b>CLIP Zero-Shot Classify</b></a><br/>
      <a href="open_vocabulary/clipResNet50/README.md">
        <img src="open_vocabulary/clipResNet50/assets/labeled_image.jpg" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Zero-shot object classification</sub><br/>
      <sub>Model: ClipResNet50</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
</table>

<a id="image-inference"></a>
### Image Inference 🖼️
Explore models performing inference on static images and data. These examples demonstrate how to leverage the MXA to process large amounts of data.

<table>
  <tr>
    <td align="center" valign="top" width="35%">
      <a href="image_inference/oriented_bounding_boxes/README.md"><b>Oriented Boxes (Satellite)</b></a><br/>
      <a href="image_inference/oriented_bounding_boxes/README.md">
        <img src="image_inference/oriented_bounding_boxes/assets/parking_lot_output.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Oriented bounding boxes on satellite imagery</sub><br/>
      <sub>Model: YoloV8m-OBB</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="35%">
      <a href="image_inference/face_recognition/README.md"><b>Face Detection + Recognition</b></a><br/>
      <a href="image_inference/face_recognition/README.md">
        <img src="image_inference/face_recognition/assets/face.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Face detection + recognition on images</sub><br/>
      <sub>Models: YoloV8n-Face &amp; FaceNet</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
</table>


<a id="multi-stream-applications"></a>
### Multi-Stream Applications 🖥️

Maximize performance by running multiple video streams concurrently on MemryX accelerators.

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <a href="multistream_video_inference/object_detection_yolov8/README.md"><b>Multi-Stream YOLOv8</b></a>
      <a href="https://developer.memryx.com/tutorials/multistream_realtime_inf/multistream_od_yolov8s.html">📝</a><br/>
      <a href="multistream_video_inference/object_detection_yolov8/README.md">
        <img src="multistream_video_inference/object_detection_yolov8/assets/yolov8_objectDetection.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Detect objects across multiple streams</sub><br/>
      <sub>Model: YOLOv8 (Small)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="33%">
      <a href="multistream_video_inference/multistream_objectdetection_yolov7Tiny/README.md"><b>Multi-Stream YOLOv7</b></a>
      <a href="https://developer.memryx.com/tutorials/multistream_realtime_inf/multistream_od.html">📝</a><br/>
      <a href="multistream_video_inference/multistream_objectdetection_yolov7Tiny/README.md">
        <img src="multistream_video_inference/multistream_objectdetection_yolov7Tiny/assets/yolov7_objectDetection_multistream.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Detect objects across multiple streams</sub><br/>
      <sub>Model: YOLOv7 (Tiny)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="33%">
      <a href="multistream_video_inference/yolo_object_detection_hw_decoding/README.md"><b>Detection with H/W Decoding</b></a><br/>
      <a href="multistream_video_inference/yolo_object_detection_hw_decoding/README.md">
        <img src="multistream_video_inference/yolo_object_detection_hw_decoding/YOLO_preview.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Multi-Stream YOLO Detection with HW-Accelerated Video Decoding</sub><br/>
      <sub>Models: YOLOv8/9/10/11</sub><br/>
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="33%">
      <a href="multistream_video_inference/yolov8_pcb_defect_detection/README.md"><b>PCB Defect Detection</b></a><br/>
      <a href="multistream_video_inference/yolov8_pcb_defect_detection/README.md">
        <img src="multistream_video_inference/yolov8_pcb_defect_detection/assets/pcb_defect.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Detect PCB defects</sub><br/>
      <sub>Model: Retrained YOLOv8s</sub><br/>
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="33%">
      <a href="multistream_video_inference/multistreamed_intrusion_detection/README.md"><b>Intrusion Detection &amp; Recognition</b></a><br/>
      <a href="multistream_video_inference/multistreamed_intrusion_detection/README.md">
        <img src="multistream_video_inference/multistreamed_intrusion_detection/assets/example.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Multi-Stream Intrusion Detection &amp; Recognition</sub><br/>
      <sub>Models: YOLOv8n-Face and FaceNet</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="33%">
      <a href="multistream_video_inference/objectDet_poseEst_yolov8/README.md"><b>Multi-Model + Multi-Stream</b></a><br/>
      <a href="multistream_video_inference/objectDet_poseEst_yolov8/README.md">
        <img src="multistream_video_inference/objectDet_poseEst_yolov8/assets/yolov8_objDet_poseEst.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Detection &amp; Pose models together in a single DFP</sub><br/>
      <sub>Models: YOLOv8n and YOLOv8n-pose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
</table>


<a id="multi-dfp-applications"></a>
### Multi-DFP Applications 🔀

Run multiple models simultaneously on separate DFPs within a single application.

<table>
  <tr>
    <td align="center" valign="top" width="50%">
      <a href="multi_dfp_application/cartoonizer_pose/README.md"><b>Cartoonizer + Pose (Side-by-Side)</b></a><br/>
      <a href="multi_dfp_application/cartoonizer_pose/README.md">
        <img src="multi_dfp_application/cartoonizer_pose/assets/cartoon_pose.png" height="115" />
      </a><br/>
      <sub>Run 2 models on separate DFPs (side-by-side)</sub><br/>
      <sub>Models: Facial-Cartoonizer &amp; YOLOv8s-pose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="C++" src="https://img.shields.io/badge/C++-blue" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="50%">
      <a href="multi_dfp_application/cartoonizer_pose_overlay/README.md"><b>Cartoonizer + Pose (Overlay)</b></a><br/>
      <a href="multi_dfp_application/cartoonizer_pose_overlay/README.md">
        <img src="multi_dfp_application/cartoonizer_pose_overlay/assets/overlay.png" height="115" />
      </a><br/>
      <sub>Overlay cartoon + pose estimation outputs</sub><br/>
      <sub>Models: Facial-Cartoonizer &amp; YOLOv8s-pose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="50%">
      <a href="multi_dfp_application/facecrop_cartoonizer/README.md"><b>Face Crop -&gt; Cartoonizer (Conditional Pipeline)</b></a><br/>
      <a href="multi_dfp_application/facecrop_cartoonizer/README.md">
        <img src="multi_dfp_application/facecrop_cartoonizer/assets/nightmare_vision.png" height="115" />
      </a><br/>
      <sub>Run DFP based on prior inference results</sub><br/>
      <sub>Models: Face Detection &amp; Facial-Cartoonizer</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"  src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="50%">&nbsp;</td>
  </tr>
</table>


<a id="fun-projects"></a>
### Fun Projects 🤖
Explore interactive and engaging AI-powered applications in our **fun projects** section.

<table>
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/chrome_dino_game/README.md"><b>Chrome Dino Game</b></a>
      <a href="https://developer.memryx.com/tutorials/fun_projects/dino_game.html">📝</a><br/>
      <a href="fun_projects/chrome_dino_game/README.md">
        <img src="fun_projects/chrome_dino_game/assets/dino_game.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Control Chrome Dino with palm</sub><br/>
      <sub>Model: Palm Detection</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/mario_rl/README.md"><b>Mario RL</b></a><br/>
      <a href="fun_projects/mario_rl/README.md">
        <img src="fun_projects/mario_rl/assets/mario.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Play Mario with a RL agent</sub><br/>
      <sub>Model: Custom</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/aimbot/README.md"><b>Aimbot</b></a><br/>
      <a href="fun_projects/aimbot/README.md">
        <img src="fun_projects/aimbot/assets/aimbot_demo.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Auto aim + click demo (Windows)</sub><br/>
      <sub>Model: YOLOv7 (Tiny)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/MxFit/README.md"><b>MxFit Repcounting</b></a><br/>
      <a href="fun_projects/MxFit/README.md">
        <img src="fun_projects/MxFit/assets/MxFit.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Workout rep-counting web app</sub><br/>
      <sub>Model: YOLOv8 Pose (medium)</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/cartoonizer/README.md"><b>Facial Cartoonizer</b></a><br/>
      <a href="fun_projects/cartoonizer/README.md">
        <img src="fun_projects/cartoonizer/assets/cartoonizer.png" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Instant cartoonize videos in real time</sub><br/>
      <sub>Model: Facial-Cartoonizer</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <img alt="Windows" src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="16" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/virtual_painter/README.md"><b>Virtual Painter</b></a><br/>
      <a href="fun_projects/virtual_painter/README.md">
        <img src="fun_projects/virtual_painter/assets/virtual_painter.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Paint virtually with hand landmarks</sub><br/>
      <sub>Models: Palm Detection &amp; Hand LM</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/asl_alphabet/README.md"><b>ASL Alphabet to Text</b></a><br/>
      <a href="fun_projects/asl_alphabet/README.md">
        <img src="fun_projects/asl_alphabet/assets/asl_demo.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Spell words using ASL</sub><br/>
      <sub>Models: Palm Detection &amp; Hand LM</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/handPong_game/README.md"><b>Hand Pong Game</b></a><br/>
      <a href="fun_projects/handPong_game/README.md">
        <img src="fun_projects/handPong_game/assets/PONG_DEMO.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Classic Pong controlled by hand tracking</sub><br/>
      <sub>Model: PalmDet &amp; HandPose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
  </tr>
  <!-- new row -->
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/gesture_web_control/README.md"><b>Gesture Scrolling</b></a><br/>
      <a href="fun_projects/gesture_web_control/README.md">
        <img src="fun_projects/gesture_web_control/assets/sample.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Gesture-controlled web scrolling</sub><br/>
      <sub>Model: YOLOv8m-pose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
      <sub>(Only X11)</sub>
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">
      <a href="fun_projects/fitness_mirror_yolov8/README.md"><b>Fitness Mirror</b></a><br/>
      <a href="fun_projects/fitness_mirror_yolov8/README.md">
        <img src="fun_projects/fitness_mirror_yolov8/assets/fitness_mirror.gif" style="height:165px; object-fit:cover;" />
      </a><br/>
      <sub>Real-time pose comparison &amp; scoring</sub><br/>
      <sub>Model: YOLOv8m-Pose</sub><br/>
      <img alt="Python" src="https://img.shields.io/badge/Python-green" />
      <img alt="Linux"   src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20" />
    </td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">&nbsp;</td>
    <!-- new cell -->
    <td align="center" valign="top" width="25%">&nbsp;</td>
  </tr>
</table>



<a id="audio-inference"></a>
### Audio Inference 🔊

Explore how the MemryX accelerators can be used for audio/speech processing tasks. The examples below demonstrate how to run models using MemryX hardware.

<div style="width: 100%;">

| Task                           | Description                                | Models  | Code             | OS  |
|--------------------------------|--------------------------------------------|---------|------------------|-----|
| [**Audio Denoising using UNet**](audio_inference/audio_denoising_cmd/README.md) | Remove noise from speech audio using UNet | UNet    | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20"> |
| [**Audio Classification using YAMNet**](audio_inference/audio_classification_cmd/README.md) | Identify audio categories using YAMNet | YAMNet  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20"> |
| [**Audio Classification Web App**](audio_inference/audio_classification_web/README.md) | Classify audio using YAMNet in a web app | YAMNet  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20"> |
| [**Speech Emotion Recognition Web App**](audio_inference/speech_emotion_recognition/README.md) | Detect emotion from speech (web app) | Light-SERNet  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20"> |


<a id="accuracy-calculation"></a>
### Accuracy Calculation ✅
Measure and evaluate the accuracy of various models using MemryX hardware.

<div style="width: 100%;">

| Task                           | Description                                       | Models           | Code                   | OS |
|--------------------------------|---------------------------------------------------|------------------|------------------------|----|
| [**Classification Accuracy**](accuracy_calculation/classification_resnet50/README.md) [📝](https://developer.memryx.com/accuracy/mlperf_accuracy/resnet50v1.5_mlperf_accuracy.html) | Calculate accuracy for classification models      | ResNet50         | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20">
| [**Object Detection Accuracy**](accuracy_calculation/detect_yolov8/README.md) [📝](https://developer.memryx.com/tutorials/accuracy/yolov8_accuracy/yolov8_accuracy.html) | Calculate accuracy for object detection models    | YOLOv8 (Medium)  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20">
| [**Keras Classifiers Accuracy**](accuracy_calculation/keras_accuracy/README.md) [📝](https://developer.memryx.com/tutorials/accuracy/keras_classifiers_accuracy/keras_accuracy_rst.html) | Calculate Keras classifiers accuracy on the MXA    | Keras applications  | ![python-badge] | <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg"  width="20" height="20">


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
