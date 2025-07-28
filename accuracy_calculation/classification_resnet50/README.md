# ResNet50 Classification Accuracy Calculation

The **ResNet50 Classification** example demonstrates how to validate the accuracy of the pretrained ResNet50 model on the MemryX accelerator. This guide provides setup instructions, model details, and code snippets to help you get started quickly.

## Overview

| **Property**         | **Details**                                                                                              |
|----------------------|-----------------------------------------------------------------------------------------------------------
| **Model**            | [ResNet50](https://docs.mlcommons.org/inference/benchmarks/image_classification/resnet50/#__tabbed_3_2)  |
| **Model Type**       | Classification                                                                                           |      
| **Framework**        | [TensorFlow](https://www.tensorflow.org/)                                                                |
| **Model Source**     | [resnet50_v1.pb](https://zenodo.org/record/2535873/files/resnet50_v1.pb) (Downloaded Automatically)      |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/2p0/mlperf_accuracycalc_resnet50_v1.zip)                                                                  |  
| **Input**            | 224x224x3                                                                                                |  
| **Output**           | class probabilities(Softmax) and class with highest confidence(Argmax)                                   |
| **License**          | [MIT](LICENSE.md)                                                                                        |

## Requirements


Before running the application, ensure that Python, and the `cv2` package are installed:

```bash
pip install opencv-python==4.11.0.86
```

## Running the Application (Linux)

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, and use the following commands:
```bash
wget https://developer.memryx.com/example_files/2p0/mlperf_accuracycalc_resnet50_v1.zip
mkdir -p models
unzip mlperf_accuracycalc_resnet50_v1.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained resnet model
 
```bash
wget https://zenodo.org/record/2535873/files/resnet50_v1.pb -O resnet50_v1.pb 
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
cd models
mx_nc -v -m resnet50_v1.pb --autocrop -c 4
```

</details>

Your folder structure should now be:

```
|- README.md
|- LICENSE.md
|- assets/
|  |-ImageNet2012_valdata
|     |-ground_truth.txt 
|- models/   
|  |- resnet50_v1.dfp
|  |- resnet50_v1.pb
|
|- src/
|  |- python 
|      |- get_imagenet_valdata.sh
|      |- preprocess.py
|      |- run_validation.py

```


### Step 2: Run the Script/Program 

The application is contained in `src/python/run_validation.py`. It takes two optional arguments:
* `--device` or `-d` - Specifies the device to run the validation on (mxa, cpu). Default is 'mxa'.
* `--count` or `-c`  - Specifies the number of images from the ImageNet2012 to run validation on. Default is 50000.

The model (~98 MB) and ImageNet dataset (~6.8 GB) are downloaded automatically if not found locally. If a pre-compiled DFP are not found, then the model is automatically compiled to generate these. Refer to the `compile_model` function to see how this is done.


To run the application:

```bash
cd src/python && python run_validation.py  # Runs on MXA
```

Other ways to run:
```bash
cd src/python && python run_validation.py --device cpu   # Runs on cpu (establish baseline performance)
cd src/python && python run_validation.py --count 10000  # Runs accuracy for 10000 images instead of 50000 images
```


## Tutorial

A more detailed tutorial with a complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/tutorials/accuracy/mlperf_accuracy/resnet50v1.5_mlperf_accuracy.html)


## Third-Party License

This project uses third-party software and libraries. Below are the details of the licenses for these dependencies:

- **Model**: Copyright (c) [Soumith Chintala]
  - License: [BSD 3-Clause](https://github.com/pytorch/vision/blob/v0.8.2/LICENSE) 🔗
- **Code Source**: The code implementation for preprocessing is based on [MLCommons Repository](https://github.com/mlcommons) 🔗
  - License: [Apache License 2.0](https://github.com/mlcommons/inference/blob/master/LICENSE.md) 🔗

## Summary

This guide offers a quick and easy way to validate the accuracy of a pretrained ResNet50 model on the ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) dataset using MemryX accelerator. You can use the provided Python script to replicate the MLPerf Benchmark Accuracy with MXA chip. Download the full code and the pre-compiled DFP file to get started immediately.
