# Audio Denoising using UNet

The **Audio Denoising using UNet** example demonstrates how to denoise an audio clip (.wav file) and listen to the enhanced audio. In particular, the focus will be on noisy speech signals. The entire process uses the MemryX accelerator to perform inference. This guide provides information about how to setup files, information about the model and code snippets to help you quickly get started.


## Overview

<div style="display: flex">
<div style="">

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | [Audio Denoising](https://github.com/vbelz/Speech-enhancement/tree/master?tab=readme-ov-file)
| **Model Type**       | Enhancement
| **Framework**        | [Keras](https://www.tensorflow.org/)
| **Model Source**     | [Download from Github](https://github.com/vbelz/Speech-enhancement/tree/master?tab=readme-ov-file)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/Audio_Denoising_UNet_128_128_1_keras.zip)
| **Input**            | Audio clips (.wav files)
| **Output**           | Denoised audio file
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)



## Requirements

Before running the application, ensure that **numpy**, **librosa** and **soundfile** are installed. You can install using the following commands:

```bash
pip install numpy librosa==0.11.0 soundfile==0.13.1
```


## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/Audio_Denoising_UNet_128_128_1_keras.zip
mkdir -p models
unzip Audio_Denoising_UNet_128_128_1_keras.zip -d models
```


<details> 
<summary> (Optional) Download and compile model yourself </summary>

Since the trained model is not directly available for download, the model was trained using the instructions given [here](https://github.com/vbelz/Speech-enhancement/tree/master?tab=readme-ov-file). 

Once you have trained the model, navigate back to the project folder and follow the steps below:

```bash
cd audio_denoising_cmd

mkdir models 
cd models 

```

Move the model into this folder and rename it to **Audio_Denoising_UNet_128_128_1_keras.h5**.

Now you may compile the model. Run the following command to generate the DFP. 


```bash
 mx_nc Audio_Denoising_UNet_128_128_1_keras.h5 -v 
```

This completes the process of download and compilation. 
</details>



### Step 2: Running the Script/Program

With the compiled model, you can now choose to provide any audio file via the command line and get the audio after the noise has been removed. This is how you can do it in Python.

#### Python

To run the example on the MX3, simply execute the following command:

1. Use the default noisy audio file:

```bash
cd src/python
python prediction_denoise.py 
```

2. Provide a path to the noisy audio file:
```bash
cd src/python
python prediction_denoise.py --path_to_noisy_audio_file <mention path here >
```

3. Provide a path to the noisy audio file and specify the path where to save the denoised audio file:
```bash
cd src/python
python prediction_denoise.py --path_to_noisy_audio_file <mention path here > --path_to_save_denoised_audio_file <mention path here >
```


## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [UNet-Keras] The UNet model with modifications for the task of denoising is provided [here](https://github.com/vbelz/Speech-enhancement/tree/master?tab=readme-ov-file) 🔗  
  - License: [MIT](https://github.com/vbelz/Speech-enhancement/blob/master/LICENSE) 🔗

- **Code Reuse**: Some code components, including pre/post-processing, were sourced from the demo code provided on [Speech enhancement](https://github.com/vbelz/Speech-enhancement/tree/master?tab=readme-ov-file) 🔗  
  - License: [MIT](https://github.com/vbelz/Speech-enhancement/blob/master/LICENSE) 🔗

- **Sample audio**: The test audio clips linked to this example were taken from the referenced websites. [Speech enhancement](https://github.com/vbelz/Speech-enhancement/tree/master?tab=readme-ov-file) 🔗  and [SpEAR](https://github.com/dingzeyuli/SpEAR-speech-database)
  - License: [MIT](https://github.com/vbelz/Speech-enhancement/blob/master/LICENSE) 🔗


## Summary

This guide offers a quick and easy way to run audio denoising from the command line on MemryX accelerators. Go ahead and download the full code to get started now!
