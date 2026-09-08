import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import CompactCNN
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "compact_cnn_best.pt"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_wafer_dataset.npz"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict_from_checkpoint(checkpoint):
    # handles a checkpoint saved as a raw state_dict or wrapped in a dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    
    return checkpoint


def save_classification_report(y_true, y_pred, class_names, out_path):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    accuracy = report_dict.pop('accuracy')

    df = pd.DataFrame(report_dict).T
    df.loc['accuracy', 'f1-score'] = accuracy
    df.loc['accuracy', 'support'] = df.loc['macro avg', 'support']

    df.to_csv(out_path)
    return df, accuracy


def save_confusion_matrix(y_true, y_pred, class_names, csv_path, png_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(csv_path)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax,)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - Compact CNN (Test Set)')
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close(fig)

    return cm_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # load test data
    data = np.load(DATA_PATH, allow_pickle=True)
    class_names = data['class_names'].tolist()

    X_test = np.transpose(data['X_test'], (0, 3, 1, 2))  # NHWC to NCHW
    y_test = np.argmax(data['y_test'], axis=1)

    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    # load model
    # weights_only=False because PyTorch 2.6+ blocks non tensor objects by default
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = load_state_dict_from_checkpoint(checkpoint)

    model = CompactCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # run on the test set
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    # accuracy, per class precision/recall/F1, macro F1
    accuracy = accuracy_score(y_test, y_pred)
    report_df, report_accuracy = save_classification_report(y_test, y_pred, class_names, os.path.join(RESULTS_DIR, "evaluation_metrics.csv"))

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro-F1:      {report_df.loc['macro avg', 'f1-score']:.4f}")
    print()
    print(report_df.round(3).to_string())

    # confusion matrix
    cm_df = save_confusion_matrix(
        y_test, y_pred, class_names,
        os.path.join(RESULTS_DIR, "confusion_matrix.csv"),
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )
    
    print()
    print("Confusion matrix:")
    print(cm_df.to_string())

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()