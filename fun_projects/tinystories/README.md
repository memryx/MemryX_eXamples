# TinyStories

The TinyStories example showcases basic text generation on the MemryX MX3, by generating short stories appropriate for young children using a Small Language Model (SLM). This guide provides setup instructions, model details, and code snippets to run this application.

<p align="center">
  <img src="assets/tiny_stories.gif" alt="Tiny Stories Example" width="41%" />
</p>

## Overview

| **Property**         | **Details**                                                                                  
|----------------------|------------------------------------------
| **Model**            | TinyStories-33M
| **Model Type**       | Text Generation
| **Framework**        | ONNX
| **Model Source**     | [HuggingFace](https://huggingface.co/roneneldan/TinyStories-33M)
| **Pre-compiled DFP** | [tinystories-33M.tar.xz](https://developer.memryx.com/example_files/tinystories-33M.tar.xz)
| **DevHub Tutorial**  | [Link](https://developer.memryx.com/tutorials/text_generation/tinystories.html)
| **Input**            | Text (up to 128 tokens long)
| **Output**           | Text
| **OS**               | Linux
| **License**          | [MIT](LICENSE.md)

## Requirements

Before running, please make sure to install the HuggingFace `transformers` package in your Python venv:

```bash
pip install transformers
```

This package contains the tokenizers, etc., used in generative language models.

Also, To run and deploy the application on the web, install Streamlit:

```bash
pip install streamlit
```

## Running the Application

### Step 1: Download the Model

In the folder for this tutorial, run:

```bash
wget https://developer.memryx.com/example_files/tinystories-33M.tar.xz
tar -xvf tinystories-33M.tar.xz
```

The necessary files should now be in the `models/` directory.


<details>
<summary>(Optional) Alternatively, Download & Compile Yourself</summary>
<br>

The following content is part of the [TinyStories DevHub Tutorial](https://developer.memryx.com/tutorials/text_generation/tinystories.html). Please read that tutorial for deeper details and code explanations.

Run the export script from the DevHub tutorial:

```bash
python3 src/export_to_onnx.py
```

Then compile the Model:

```bash
cd models &&
mx_nc -v -m tinystories33M.onnx --inputs /original_model/transformer/Add_output_0:0,/original_model/transformer/Add_output_0:1,/original_model/transformer/Add_output_0:2 --outputs /original_model/transformer/ln_f/Add_1_output_0 --graph_extensions TinyStories --insert_onnx_io_format_adapters "io" --effort hard &&
cd -
```

**IMPORTANT**: This commands uses the *"Graph Extensions"* feature coming in the [SDK 1.1 release](https://developer.memryx.com/support/roadmap.html). If you're on SDK 1.0, please download the precompiled version from above instead.
</details>


### Step 2: Running

Simply run:

```bash
streamlit run run_tinystories.py
```

You will be presented with a prompt to start your story, then the language model will complete it.

```bash
Type the beginning of a story: [type here and hit ENTER]
```

The length of the story will vary, as we let the language model run until it hits an "End-of-Story" token.

Also note that this is a very small model that writes simple children's stories, so it cannot do question answering, etc. like a full LLM. Instead, the purpose of this model is to showcase how small language models may become in the future.


## Tutorial

See the complete tutorial [on the DevHub](https://developer.memryx.com/tutorials/text_generation/tinystories.html).

## Third-Party License

This project uses third-party software and libraries. Below are the details of the licenses for these dependencies:

* Model: Copyright (c) Ronen Eldan, [MIT license](https://huggingface.co/roneneldan/TinyStories-33M/) 🔗
