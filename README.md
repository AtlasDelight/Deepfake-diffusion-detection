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


## Compile LaTeX reports and slides locally with VS Code

This project contains LaTeX reports and Beamer slides inside the `reports/` folder.

Instead of using Overleaf, the `.tex` files can be compiled locally with VS Code.

### 1. Install LaTeX tools

On Debian / Ubuntu / WSL, install a LaTeX distribution and `latexmk`:

```bash
sudo apt update
sudo apt install texlive-full latexmk
```

If `texlive-full` is too large, a lighter installation can be used:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-bibtex-extra biber latexmk
```

### 2. Install the VS Code extension

In VS Code, install the extension:

```text
LaTeX Workshop
```

This extension allows VS Code to compile `.tex` files and preview the generated PDF.

### 3. Open the project in VS Code

From the project root:

```bash
code .
```

Then open the LaTeX file you want to compile, for example:

```text
reports/SoTA/Detection_deepfakes_slides/main.tex
```

or:

```text
reports/SoTA/Detrction_deepfakes/main.tex
```

### 4. Compile from VS Code

With `main.tex` open in VS Code, use:

```text
Ctrl + Alt + B
```

or open the Command Palette:

```text
Ctrl + Shift + P
```

then search for:

```text
LaTeX Workshop: Build LaTeX project
```

The generated PDF will appear in the same folder as `main.tex`.

### 5. Compile from the terminal

You can also compile manually from the terminal.

Example for the Beamer slides:

```bash
cd reports/SoTA/Detection_deepfakes_slides
latexmk -pdf -interaction=nonstopmode -synctex=1 main.tex
```

Example for the written report:

```bash
cd reports/SoTA/Detrction_deepfakes
latexmk -pdf -interaction=nonstopmode -synctex=1 main.tex
```

To clean temporary LaTeX files:

```bash
latexmk -c
```

To remove all generated files, including the PDF:

```bash
latexmk -C
```

### 6. Recommended VS Code settings

Create the file:

```text
.vscode/settings.json
```

and add:

```json
{
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.latex.outDir": "%DIR%",
  "latex-workshop.view.pdf.viewer": "tab",
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": ["latexmk"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-pdf",
        "-interaction=nonstopmode",
        "-synctex=1",
        "%DOC%"
      ]
    }
  ]
}
```

With this configuration, saving a `.tex` file automatically rebuilds the PDF.

### 7. Git ignore for LaTeX temporary files

LaTeX generates many temporary files during compilation. These files should not be committed.

Add the following lines to `.gitignore`:

```gitignore
# LaTeX temporary files
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.synctex.gz
*.nav
*.snm
*.vrb
*.lof
*.lot
*.bcf
*.run.xml
```

The `.tex`, `.bib`, images, and final PDF can be committed if needed.