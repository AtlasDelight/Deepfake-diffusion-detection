# Deepfake Diffusion Detection

This project explores lightweight adaptations of diffusion-based deepfake detection methods.

The goal is not to directly reproduce the official heavy implementations of methods such as DIRE, LaRE², DNF, SeDID or related approaches. Instead, this project studies whether their core ideas can be adapted using a smaller pretrained generative model.

The main idea is:

```text
image x
  -> lightweight pretrained generative model
  -> reconstruction / latent / noise-based signal
  -> error map
  -> lightweight classifier
  -> real / fake prediction
```


## Current pretrained model
The current lightweight generative model used in this project is:
```bash
T2I-ImageNet
https://github.com/lucasdegeorge/T2I-ImageNet
This repository is cloned locally into:
external/T2I-ImageNet/
```

## Setup instructions
Some folders and files are intentionally ignored by Git because they are too large, generated automatically, or specific to each local machine.

Ignored folders typically include:
```text
.venv/
external/
data/
datasets/
weights/
checkpoints/
error_maps/
outputs/
generated_maps/
```
To clone repository:
```bash
git clone https://github.com/AtlasDelight/deepfake-diffusion-detection.git
cd deepfake-diffusion-detection
```
To create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```
(After activation, the terminal should show (.venv) at the beginning of the command line.)


To Install dependencies:
```bash
python3 -m pip install --upgrade pip wheel
python3 -m pip install "setuptools<70"
python3 -m pip install -r requirements.txt
```
To clone the external T2I-ImageNet repository:
The external model repository is not stored inside this GitHub project.
you have to clone it manually:

```bash
mkdir -p external
git clone https://github.com/lucasdegeorge/T2I-ImageNet.git external/T2I-ImageNet
```
Then install it in editable mode:
``` bash
python3 -m pip install -e external/T2I-ImageNet --no-build-isolation
```
Hugging Face authentication:
Some model components may require Hugging Face access. Install the Hugging Face tools:
```bash
python3 -m pip install -U huggingface_hub accelerate
```

Then log in:
```bash
hf auth login
```
If the hf command is not available, use:
```bash
huggingface-cli login
```
A Hugging Face access token with read permissions is required.