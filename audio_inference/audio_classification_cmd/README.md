# Audio Classification using YAMNet

The **Audio classification using YAMNet** example demonstrates how to classify audio inputs using the YAMNet model on MemryX accelerators. This guide provides information about how to setup files, information about the model and code snippets to help you quickly get started.


## Overview

<div style="display: flex">
<div style="">

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | [Audio classification(YAMNet)](https://www.kaggle.com/models/google/yamnet/tfLite)
| **Model Type**       | Classification
| **Framework**        | [Tflite](https://www.tensorflow.org/)
| **Model Source**     | [Download from Kaggle](https://www.kaggle.com/models/google/yamnet/tfLite)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p0/Audio_classification_YamNet_96_64_1_tflite.zip)
| **Input**            | Audio clips (.wav files)
| **Output**           | Class to which the audio clip mostly is about.
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)



## Requirements

Before running the application, ensure that **ai_edge_litert**, **flask** and **scipy** are installed. You can install using the following commands:

```bash
pip install ai-edge-litert==1.3.0 scipy flask==3.1.1
```

NOTE: The package **ai-edge-litert** is only supported in Python versions 3.9 - 3.12. Please make sure you have right versions of Python installed.


## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/model_explorer/2p0/Audio_classification_YamNet_96_64_1_tflite.zip
mkdir -p models
unzip Audio_classification_YamNet_96_64_1_tflite.zip -d models
```

<details> 
<summary> (Optional) Download and compile model yourself </summary>

First, let us create the folders required to store the model using the commands below:

```bash
cd audio_classification_cmd

mkdir models 
cd models 

```

Follow the steps below to download the model. 

```bash
curl -L -o ./model.tar.gz https://www.kaggle.com/api/v1/models/google/yamnet/tfLite/tflite/1/download
tar -xzf ./model.tar.gz -C ./
mv 1.tflite Audio_classification_YamNet_96_64_1_tflite.tflite

```

Now you may compile the model. Run the following command to generate the DFP. 

```bash
 mx_nc Audio_classification_YamNet_96_64_1_tflite.tflite -v --autocrop
```

This completes the process of download and compilation. 
</details>


### Step 2: Running the Script/Program

With the compiled model, you can now choose to provide any audio file via the command line and see what the main object of inference is. This is how you can do it in Python.

#### Python

To run the example on the MX3, simply execute the following command:

```bash
cd src/python
python classify_audio.py --audio_file_path <path_to_the_audio_file>
```

For example:
```bash
python classify_audio.py --audio_file_path '../../assets/44737-5-0-1.wav'
```

If no audio file path is provided, it will use the default audio file provided. Once the above command is executed, the predicted class will be displayed in the output console.

```bash
Sample rate: 16000 Hz
Total duration: 4.00s
The predicted class is: Vehicle
```


And that's it!! Classification is now complete and the predicted class is displayed.

## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [YAMNet Model (TFLite)](https://www.kaggle.com/models/google/yamnet/tfLite) 🔗  
  - License: [Apache-2.0](https://developers.google.com/terms/site-policies) 🔗

- **Code Reuse**: Some code components, including pre/post-processing, were sourced from the demo code provided on [Yamnet-Kaggle](https://www.kaggle.com/models/google/yamnet/tfLite) 🔗  
  - License: [Apache-2.0](https://www.tensorflow.org/hub/tutorials/yamnet) 🔗

- **Sample audio**: The test audio clips linked to this example were taken from the UrbanSound8K dataset. [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) 🔗  
  - License: [Creative Commons Attribution Noncommercial License, version 3.0](https://urbansounddataset.weebly.com/) 🔗


## Summary

This guide offers a quick and easy way to run audio classification from the command line using the YAMNet model on MemryX accelerators. Go ahead and download the full code to get started now!
