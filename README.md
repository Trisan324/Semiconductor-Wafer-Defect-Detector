# ENEL4AI Semiconductor Wafer Defect Detection

## Project Overview

This group project investigates the use of artificial intelligence to classify semiconductor wafer defect patterns. The project will use the WM-811K Wafer Map dataset and a supervised Convolutional Neural Network (CNN).

To keep the project achievable within the four-week development period, the model will focus on a selected subset of five to six common wafer-map classes.

## Project Objective

The objective is to develop and evaluate an image-classification model that accepts a semiconductor wafer map as input and predicts its corresponding defect-pattern class.

## Dataset

The project uses the WM-811K Wafer Map dataset, which contains wafer maps collected from semiconductor fabrication processes.

Dataset source:

* [WM-811K Wafer Map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
* [Original WM-811K publication](https://doi.org/10.1109/TSM.2014.2364237)

The dataset is not included in this repository because of its size. Each group member must download it separately and place it in the local `data/raw/` directory.

## Technology Stack

* Python
* PyTorch
* Visual Studio Code
* Jupyter Notebook
* NumPy
* pandas
* scikit-learn
* OpenCV
* Matplotlib
* Seaborn
* Git and GitHub

Model training will primarily be performed locally on a GPU-enabled computer.

## Repository Structure

```text
data/
├── raw/             Original WM-811K dataset
└── processed/       Cleaned and preprocessed data

notebooks/           Data exploration and model experiments
src/                 Reusable Python source code
models/              Saved model files
results/             Evaluation results, graphs and confusion matrices

README.md            Project information and setup instructions
requirements.txt     Required Python packages
.gitignore           Files excluded from version control
```

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/ENEL4AI-Wafer-Defect-Detection.git
cd ENEL4AI-Wafer-Defect-Detection
```

Create a Python virtual environment on Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the core dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch must be installed separately using the appropriate command for the computer’s operating system and GPU:

* [PyTorch installation selector](https://pytorch.org/get-started/locally/)
