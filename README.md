# Semiconductor Wafer Defect Detection

## Project Overview

This group project uses artificial intelligence to classify semiconductor wafer defect patterns. It uses the WM-811K Wafer Map dataset and a compact Convolutional Neural Network (CNN) built in PyTorch.

The project focuses on six wafer-map classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, and Scratch. A baseline CNN has been trained and evaluated on this subset, and the trained model and results are included in this repository.

## Project Objective

The objective is to develop and evaluate an image-classification model that accepts a semiconductor wafer map as input and predicts its corresponding defect-pattern class.

## Dataset

The project uses the WM-811K Wafer Map dataset, which contains wafer maps collected from semiconductor fabrication processes.

Dataset sources:

- [WM-811K Wafer Map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- [Original WM-811K publication](https://doi.org/10.1109/TSM.2014.2364237)

The raw dataset (`LSWMD.pkl`, about 2 GB) is not included in this repository because of its size. Anyone who needs to rerun preprocessing from scratch should download it from Kaggle and place it in `data/raw/`.

The processed dataset (`data/processed/processed_wafer_dataset.npz`) is included, since it is small and lets anyone run training or evaluation immediately without needing the raw file.

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

Model training was performed locally. GPU acceleration is used where supported, with CPU training available as a fallback. The primary training computer uses an AMD Radeon RX 6600 through PyTorch DirectML.

## Repository Structure

```text
data/
├── raw/                            Original WM-811K dataset (not included, download separately)
└── processed/                      Processed dataset used for training and evaluation

notebooks/                          Reserved for exploratory work
src/
├── data_acquisition_cleaning.py    Builds the processed dataset from the raw WM-811K file
├── inspect_npz.py                  Prints the arrays, shapes, and dtypes in the processed dataset
├── check_directml.py               Confirms PyTorch can see the AMD GPU through DirectML
├── model.py                        CompactCNN architecture
├── train.py                        Trains the model and saves the best checkpoint and training log
├── evaluate.py                     Evaluates the trained model on the test set
└── plot_results.py                 Generates the report figures from the training log and evaluation output

models/                             Saved model checkpoint
results/                            Training log, evaluation metrics, confusion matrix, and figures

README.md                           Project information and setup instructions
requirements.txt                    Shared Python dependencies
requirements-amd-directml.txt       AMD DirectML environment
.gitignore                          Files excluded from version control
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

## How to Run

Run these from the project root, with the virtual environment activated.

Build the processed dataset from the raw WM-811K file. Skip this step if you are using the processed dataset already included in the repository.

```powershell
python src/data_acquisition_cleaning.py
```

Train the model. Saves the best checkpoint to `models/compact_cnn_best.pt` and the training log to `results/training_log.csv`.

```powershell
python src/train.py
```

Evaluate the trained model on the test set. Saves accuracy, per-class precision, recall, F1-score, and the confusion matrix to `results/`.

```powershell
python src/evaluate.py
```

Generate the report figures from the training log and evaluation output. Saves to `results/figures/`.

```powershell
python src/plot_results.py
```

## Results

The baseline CompactCNN reached 86.66% validation accuracy during training. On the held-out test set it achieved 85.71% accuracy and 85.21% macro-F1 across the six classes. Full per-class metrics and the confusion matrix are in `results/evaluation_metrics.csv` and `results/confusion_matrix.csv`.
