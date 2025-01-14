# Speech Emotion Recognition Web Application

The **Speech Emotion Recognition web application** example demonstrates how to detect emotion from speech inputs on MemryX accelerators. This guide provides information about how to setup files, information about the model and code snippets to help you quickly get started.


## Overview

<div style="display: flex">
<div style="">

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | [Speech Emotion Recognition](https://github.com/AryaAftab/LIGHT-SERNET/tree/master)
| **Model Type**       | Classification
| **Framework**        | [Tflite](https://www.tensorflow.org/)
| **Model Source**     | [Download from Github](https://github.com/AryaAftab/LIGHT-SERNET/tree/master)
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/1p1/speech_emotion_recognition.zip)
| **Input**            | Audio clips (.wav files)
| **Output**           | Emotion detected from the speech input
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)



## Requirements

Before running the application, ensure that **ffmpeg** and **flask** are installed. You can install using the following commands:

```bash
pip install flask
sudo apt install ffmpeg
```

NOTE: Depending on which OS you use, you would have to download the corresponding version of ffmpeg. Please refer to this [link](https://ffmpeg.org/download.html) for more details. The above command is for Linux systems.

If the above command sudo apt install ffmpeg does not work, try the following command:

```bash
pip install ffmpeg-python
```


## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/1p1/speech_emotion_recognition.zip
mkdir -p models
unzip speech_emotion_recognition.zip -d models
```


### Step 2: Running the Script/Program

With the compiled model downloaded, you can now upload / record audio and infer the emotion. This is how to do it in Python.

#### Python

To run the example on the MX3, simply execute the following command:

```bash
cd speech_emotion_recognition/src/python
python app.py 
```

After running the above commands, the link to the localhost will be displayed in the console. Follow the link to launch the web page. In the example below, it is **http://127.0.0.1:5000**.


</div>
<div style="padding-left: 100px;">
    <img src="assets/linktowebpage.png" alt="web link" style="height: 240px;">
</div>
</div>


### Step 3: Providing audio input

There are 2 ways to provide the audio input:

1. **Via file upload (.wav files only):**

You can upload a file via the **Choose File** option and then press the **Upload** button to perform inference on the MXA after which the result will be displayed in the Classification Result Box below. Additional information inferred from the audio signal is also displayed in the pie chart.

</div>
<div style="padding-left: 100px;">
    <img src="assets/webpage1.png" alt="web link" style="height: 300px;">
</div>
</div>


2. **Record audio:**

If you wish to record the audio, you may do so by selecting the **Start Recording** button. Once the recorder is initialized, select the **Start Recording** button again to now record the audio. Press the **Stop Recording** button to stop. You can also hear the recorded audio by playing the generated clip. Once the recording is complete, upload it for inferencing by selecting the **Upload Recording** button. The MXA will analyze the audio input and display the result in the Classification Result box. 

</div>
<div style="padding-left: 100px;">
    <img src="assets/recordpage.png" alt="web link" style="height: 300px;">
</div>
</div>


And that's it!! You have now completed detecting emotion from speech using the MemryX accelerators!


## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [Github](https://github.com/AryaAftab/LIGHT-SERNET/tree/master) 🔗  
  
- **Code Reuse**: Some code components, including pre/post-processing, were sourced from the demo code provided on [LIGHT-SERNET](https://github.com/AryaAftab/LIGHT-SERNET/tree/master) 🔗  

- **Sample audio**: The test audio clips linked to this example were taken from the EMODB and RAVDESS datasets. [EMO-DB](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb?resource=download) 🔗 and [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) 



## Summary

This guide offers a quick and easy way to perform speech emotion recognition on MemryX accelerators. Go ahead and download the full code to get started now!
