# Audio Classification Web Application

The **Audio classification web application (YAMNet model)** example demonstrates how to classify audio inputs using the YAMNet model on MemryX accelerators. This guide provides information about how to setup files, information about the model and code snippets to help you quickly get started.


## Overview

<div style="display: flex">
<div style="">

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | [Audio classification](https://www.kaggle.com/models/google/yamnet/tfLite)
| **Model Type**       | Classification
| **Framework**        | [Tflite](https://www.tensorflow.org/)
| **Model Source**     | [Download from Kaggle](https://www.kaggle.com/models/google/yamnet/tfLite)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/1p1/audio_classification.zip)
| **Input**            | Audio clips (.wav files)
| **Output**           | What the audio clip is mostly about
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)



## Requirements

Before running the application, ensure that **ai_edge_litert**, **ffmpeg**, **flask** and **scipy** are installed. You can install using the following commands:

```bash
pip install ai-edge-litert scipy flask
sudo apt install ffmpeg
```

NOTE: The package **ai-edge-litert** is only supported in Python versions 3.9 - 3.11. Please make sure you have right versions of Python installed. 

NOTE: Depending on which OS you use, you would have to download the corresponding version of ffmpeg. Please refer to this [link](https://ffmpeg.org/download.html) for more details. The above command is for Linux systems.

If the above command *sudo apt install ffmpeg* does not work, try the following command:

```bash
pip install ffmpeg-python
```


## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/1p1/audio_classification.zip
mkdir -p models
unzip audio_classification.zip -d models
```


### Step 2: Running the Script/Program

With the compiled model downloaded, you can now upload / record any audio and see what the main object of inference is. This is how to do it in Python.

#### Python

To run the example on the MX3, simply execute the following command:

```bash
cd src/python
python app.py 
```

After running the above commands, the link to the localhost will be displayed in the console. Follow the link to launch the web page. In the example below, it is **http://127.0.0.1:5000**.


</div>
<div style="padding-left: 100px;">
    <img src="assets/linktoweb.png" alt="web link" style="height: 240px;">
</div>
</div>


### Step 3: Providing audio input

You can go through the **Get Started** section to have a general idea of how to classify your audio. There are 2 ways to provide the audio input:

1. **Via file upload (.wav files only):**

You can upload a file via the **Choose File** option and then press the **Upload and classify** button to perform inference on the MXA after which the result will be displayed in the Classification Result Box below. Additionally, other sounds in the audio (background noise, occasional other sounds) are displayed in the pie chart.

</div>
<div style="padding-left: 100px;">
    <img src="assets/classifyfile.png" alt="web link" style="height: 300px;">
</div>
</div>


2. **Record audio:**

If you wish to record the audio, you may do so by selecting the **Start Recording** button. Once the recorder is initialized, select the **Start Recording** button again to now record the audio. Press the **Stop Recording** button to stop. You can also hear the recorded audio by playing the generated clip. Once the recording is complete, upload it for classification by selecting the **Upload Recording** button. The MXA will perform inference on the audio input and display the result in the Classification Result box. 

</div>
<div style="padding-left: 100px;">
    <img src="assets/classifyrecord.png" alt="web link" style="height: 300px;">
</div>
</div>


And that's it!! You have now completed classifying your audio using the MemryX accelerators!


## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [YAMNet Model (TFLite)](https://www.kaggle.com/models/google/yamnet/tfLite) 🔗  
  - License: [Apache-2.0](https://developers.google.com/terms/site-policies) 🔗

- **Code Reuse**: Some code components, including pre/post-processing, were sourced from the demo code provided on [Yamnet-Kaggle](https://www.kaggle.com/models/google/yamnet/tfLite) 🔗  
  - License: [Apache-2.0](https://www.tensorflow.org/hub/tutorials/yamnet) 🔗

- **Sample audio**: The test audio clips linked to this example were taken from the UrbanSound8K dataset. [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) 🔗  
  - License: [Creative Commons Attribution Noncommercial License, version 3.0](https://urbansounddataset.weebly.com/) 🔗


## Summary

This guide offers a quick and easy way to perform audio classification using the YAMNet model on MemryX accelerators. Go ahead and download the full code to get started now!
