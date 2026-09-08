from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_directml
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

# paths resolve from the project root so it doesn't matter where this is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "processed_wafer_dataset.npz"
BATCH_SIZE = 32

data = np.load(NPZ_PATH, allow_pickle=True)

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
class_names = data["class_names"]

print(f"Classes ({len(class_names)}): {list(class_names)}")


def make_loader(X, y_onehot, shuffle):
    # images: (N, 32, 32, 3) -> (N, 3, 32, 32), the layout PyTorch conv layers expect
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)

    # labels: one-hot (N, 6) -> class index (N,), what cross-entropy loss expects
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32).argmax(dim=1).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)

# model

NUM_CLASSES = 6
DROPOUT_RATE = 0.3


class CompactCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()

        # three conv blocks, filters increasing 32 -> 64 -> 128, each halving
        # the spatial size via max pooling (32x32 -> 16x16 -> 8x8 -> 4x4)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # global average pooling collapses each feature map to one number,
        # giving a fixed length vector regardless of spatial size
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x  # raw scores, softmax is applied by the loss function


# training loop

DEVICE = torch_directml.device()
LEARNING_RATE = 0.001
EPOCHS = 30


def get_class_weights(y_onehot):
    # balanced class weights so Donut and Scratch count for more in the loss
    y_indices = y_onehot.argmax(axis=1)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_indices,
    )
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    # one pass over loader. pass optimizer=None to evaluate without updating weights
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# save the best checkpoint and training logs

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = MODEL_DIR / "compact_cnn_best.pt"
LOG_PATH = RESULTS_DIR / "training_log.csv"

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training on device: {DEVICE}")

    class_weights = get_class_weights(y_train).to(DEVICE)
    print(f"Class weights: {class_weights.tolist()}")

    model = CompactCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    log_rows = ["epoch,train_loss,train_acc,val_loss,val_acc"]

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        log_rows.append(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}")

        # save only when validation accuracy improves so the checkpoint holds
        # the best epoch, not just the last one
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> new best (val_acc={val_acc:.4f}), saved to {CHECKPOINT_PATH}")

    LOG_PATH.write_text("\n".join(log_rows))
    print(f"\nTraining log saved to {LOG_PATH}")
    print(f"Best model (val_acc={best_val_acc:.4f}) saved to {CHECKPOINT_PATH}")