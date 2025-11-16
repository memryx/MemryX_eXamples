# Zero-Shot Classification — OpenAI CLIP ResNet-50

CLIP is pretrained on image–text pairs and can **classify without task-specific training**: you just supply natural-language class names (e.g., “a photo of a dog”, “a photo of a cat”).  This guide gives you a fast path to install dependencies, understand how the model works, and run a plain demo that predicts a label from any list of classes you provide.

<p align="center">
    <img src="assets/labeled_image.jpg " alt="ClipResNet50 Example" style="height: 300px;">
</p>


## Overview



| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [CLIP ResNet-50 ](https://arxiv.org/abs/2103.00020)                                            |
| **Model Type**       | Zero-shot image–text matching / classification                                                       |
| **Framework**        | [onnx](https://onnx.ai/)                                                   |
| **Model Source**     | [Download](https://github.com/openai/CLIP) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/ClipResNet50.zip)                                           |
| **Input**        | Any RGB image |
| **Output**       | class label                                     |
| **OS**               | Linux |
| **License**      | [MIT](https://github.com/openai/CLIP)   






## Requirements


```bash
pip3 install ftfy regex tqdm packaging
```



## Running the Application

### Step 1: Download and Prepare the Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:

```bash
wget https://developer.memryx.com/example_files/ClipResNet50.zip
mkdir -p models
unzip ClipResNet50.zip -d models
```


<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you need to compile the YOLOE model and generate the DFP file manually under models folder.

```bash
wget https://developer.memryx.com/example_files/ClipResNet50.zip
unzip ClipResNet50.zip -d models
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:
```bash
cd models/ 
mx_nc -v -m ClipResNet50.onnx --extensions ClipResNet50 --no_split_big_layers
```

</details>

### Step 2: Run the Program

With the compiled model ready, you can now perform real-time inference. By default, the program uses a predefined input image and class name list, but you are free to provide your own input image and custom class list for classification.

```bash
cd src/python
python3 demo.py  
```


## Third-Party Licenses

This example uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

* **Model/Code**: [OpenAI CLIP](https://github.com/openai/CLIP) 🔗
  * License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE) 🔗




## Summary

With CLIP RN50 you can **classify images without training**, just by providing a list of natural-language classes. Install dependencies, run the quickstart, and swap images or class prompts freely to explore zero-shot recognition. For more details and advanced usage (e.g., different backbones, feature extraction, linear probes), check the official CLIP README and notebooks. ([GitHub][1])

[1]: https://github.com/openai/CLIP "GitHub - openai/CLIP: CLIP (Contrastive Language-Image Pretraining),  Predict the most relevant text snippet given an image"
[2]: https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb?utm_source=chatgpt.com "Interacting with CLIP.ipynb - Colab"
[3]: https://openai.com/index/clip/?utm_source=chatgpt.com "CLIP: Connecting text and images - OpenAI"
