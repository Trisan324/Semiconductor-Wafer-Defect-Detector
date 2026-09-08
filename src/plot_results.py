from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAINING_LOG = RESULTS_DIR / "training_log.csv"
METRICS_CSV = RESULTS_DIR / "evaluation_metrics.csv"
CONFUSION_CSV = RESULTS_DIR / "confusion_matrix.csv"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place the corresponding project result in results/ and run again."
        )


def save_training_loss(training: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(training["epoch"], training["train_loss"], label="Training loss")
    ax.plot(training["epoch"], training["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Compact CNN Training and Validation Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_loss_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_training_accuracy(training: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(training["epoch"], training["train_acc"] * 100, label="Training accuracy")
    ax.plot(training["epoch"], training["val_acc"] * 100, label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Compact CNN Training and Validation Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_accuracy_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: pd.DataFrame) -> None:
    values = cm.to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(values)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(cm.columns)))
    ax.set_yticks(np.arange(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix - Compact CNN (Test Set)")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_per_class_f1(metrics: pd.DataFrame) -> None:
    summary_rows = {"macro avg", "weighted avg", "accuracy"}
    per_class = metrics.loc[[idx for idx in metrics.index if idx not in summary_rows]].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(per_class.index, per_class["f1-score"] * 100)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1-score (%)")
    ax.set_title("Per-Class F1-Score on the Test Set")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_f1_score.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for path in (TRAINING_LOG, METRICS_CSV, CONFUSION_CSV):
        require(path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(TRAINING_LOG)
    metrics = pd.read_csv(METRICS_CSV, index_col=0)
    confusion = pd.read_csv(CONFUSION_CSV, index_col=0)

    save_training_loss(training)
    save_training_accuracy(training)
    save_confusion_matrix(confusion)
    save_per_class_f1(metrics)

    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()