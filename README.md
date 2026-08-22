# Semiconductor Wafer Defect Detection

## Project Overview

This group project investigates the use of artificial intelligence to classify semiconductor wafer defect patterns. The project will use the WM-811K Wafer Map dataset and a supervised Convolutional Neural Network (CNN).

To keep the project achievable within the four-week development period, the model will focus on a selected subset of five to six common wafer-map classes.

## Project Objective

The objective is to develop and evaluate an image-classification model that accepts a semiconductor wafer map as input and predicts its corresponding defect-pattern class.

## Dataset

The project uses the WM-811K Wafer Map dataset, which contains wafer maps collected from semiconductor fabrication processes.

Dataset sources:

- [WM-811K Wafer Map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- [Original WM-811K publication](https://doi.org/10.1109/TSM.2014.2364237)

The dataset is not included in this repository because of its size. Each group member must download it separately and place it in the local `data/raw/` directory.

## Technology Stack

- Python
- PyTorch
- Visual Studio Code
- Jupyter Notebook
- NumPy
- pandas
- scikit-learn
- OpenCV
- Matplotlib
- Seaborn
- Git and GitHub

Model training will be performed locally. GPU acceleration will be used where supported, with CPU training available as a fallback. The primary training computer uses an AMD Radeon RX 6600 through PyTorch DirectML.

## Repository Structure

```text
data/
├── raw/                         Original WM-811K dataset
└── processed/                   Cleaned and preprocessed data

notebooks/                       Data exploration and model experiments
src/                             Reusable Python source code
models/                          Saved model files
results/                         Evaluation results, graphs and confusion matrices

README.md                        Project information and setup instructions
requirements.txt                 Shared Python dependencies
requirements-amd-directml.txt    AMD DirectML environment
.gitignore                       Files excluded from version control
```

## Environment Setup

### Clone the Repository

The repository can be cloned using GitHub Desktop:

1. Open GitHub Desktop.
2. Select **File → Clone repository**.
3. Select `Semiconductor-Wafer-Defect-Detection`.
4. Choose a local folder.
5. Click **Clone**.

### Create the Python Environment

Python 3.11 is recommended for compatibility.

Open Windows PowerShell inside the repository and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install the Shared Dependencies

For a standard CPU environment:

```powershell
pip install -r requirements.txt
```

### AMD GPU Setup on Windows

The primary training computer uses an AMD Radeon RX 6600 through Microsoft DirectML.

Install the shared packages and DirectML dependencies:

```powershell
pip install -r requirements-amd-directml.txt
```

Test the DirectML device:

```powershell
python -c "import torch, torch_directml; device=torch_directml.device(); x=torch.tensor([1.0]).to(device); print('Device:', device); print('Result:', (x+2).item())"
```

A successful result should resemble:

```text
Device: privateuseone:0
Result: 3.0
```

DirectML does not use NVIDIA CUDA. Therefore, `torch.cuda.is_available()` may return `False` even when the AMD GPU is working correctly.

### Other GPU Configurations

Members using NVIDIA GPUs or other operating systems should install PyTorch using the official installation selector:

- [PyTorch installation selector](https://pytorch.org/get-started/locally/)