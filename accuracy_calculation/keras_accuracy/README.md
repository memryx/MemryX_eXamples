# Keras Classifiers Accuracy Calculation

The **Keras Classification Accuracy** example demonstrates how to measure the top 1 / top 5 accuracy of classifcation models provided from [`keras.applications`](https://keras.io/api/applications/) on the MemryX accelerator. This guide provides setup instructions, model details, and code snippets to help you get started quickly.


## Overview

| **Property**         | **Details**                                                                                              |
|----------------------|-----------------------------------------------------------------------------------------------------------
| **Model**            | [Keras Classifiers](https://keras.io/api/applications/)  |
| **Model Type**       | Classification                                                                                           |      
| **Framework**        | [Keras](https://www.tensorflow.org/guide/keras)                                                                |
| **Model Source**     | [Keras homepage](https://keras.io/api/applications/)      |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/tutorials/accuracy/keras_classifiers_accuracy/keras_accuracy_rst.html)                                                                  |  
| **Input**            | Input shape depends on the model used for inference                                               |  
| **Output**           | Top 1% and Top 5% accuracies on either the CPU or MXA                                   |
| **License**          | [Apache License 2.0](https://github.com/keras-team/keras/blob/master/LICENSE)                                                                                        |



## Requirements

Before running the application, ensure that Python, and the `cv2` package are installed:

```bash
pip install opencv-python==4.11.0.86
```


## Running the Application (Linux)

### Step 1: Download Pre-compiled DFP

You can directly download the pre-compiled DFP from [here](https://developer.memryx.com/tutorials/accuracy/keras_classifiers_accuracy/keras_accuracy_rst.html). 


<details> 
<summary> (Optional) Download and compile the model </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. The script will take care of downloading the model, all you would have to do is provide the name of the model via the command line. Refer to the code snippet below for reference.



</details>

Your folder structure should now be:

```
|- README.md
|- LICENSE.md
|- assets/
|  |-ImageNet2012_valdata
|     |-ground_truth.txt    
|
|- src/
|  |- python 
|      |- get_imagenet_valdata.sh
|      |- keras_accuracy.py

```


### Step 2: Run the Script/Program 

The application is contained in `src/python/keras_accuracy.py`. It takes the following optional arguments:
* `--num_images` - Specifies the number of images to run inference on. By default, it will consider all images in the dataset.
* `--backend`    - Specifies the device to run inference on i.e. MXA or CPU. By default it will run on the MXA.
* `--dfp`        - Path to the downloaded DFP file.

The model and the ImageNet dataset (~6.8 GB) is downloaded automatically if not found locally. If a pre-compiled DFP is not specified, then the model is automatically downloaded and compiled to a DFP before running inference. 


To run the application:

```bash
cd src/python
python keras_accuracy.py --model_name 'MobileNet' --num_images 10  # Runs on MXA and does inference on 10 images. Model will be compiled to generate DFP

```

Other ways to run:
```bash
cd src/python
python keras_accuracy.py --model_name 'MobileNet' --num_images 10 --backend 'cpu' # Run on the CPU
python keras_accuracy.py --model_name 'MobileNet' --num_images 10 --backend 'mxa' --dfp 'MobileNet.dfp' # Skip model compilation since the path to the DFP is provided
```

NOTE:
If you do not have the dataset and have to download it separately via the script get_imagenet_valdata.sh, use the following command the first time:

```bash
chmod +x get_imagenet_valdata.sh 
```

## Tutorial

A more detailed tutorial with the complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/accuracy/keras_classifiers_accuracy/keras_accuracy_rst.html)


## Third-Party License

This project uses third-party software and libraries. Below are the details of the licenses for these dependencies:

- **Models**: The models are sourced directly from [Keras Applications](https://keras.io/api/applications/)🔗
  - [Apache License 2.0](https://github.com/keras-team/keras/blob/master/LICENSE)🔗

## Summary

This guide offers a quick and easy way to compare the performance of the MXA and the CPU on some of the most commonly used Keras classifiers using the ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) dataset. You can use the provided Python script to replicate the accuracy with the MXA chip. Download the full code file to get started immediately.
