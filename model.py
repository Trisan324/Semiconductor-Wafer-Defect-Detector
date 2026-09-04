"""
CompactCNN architecture - copied verbatim from the project's actual
training script (GitHub/Semiconductor-Wafer-Defect-Detector-main/src/train.py,
Step 3), so this is the authoritative definition, not a reconstruction.
"""

import torch.nn as nn

NUM_CLASSES = 6
DROPOUT_RATE = 0.3


class CompactCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()

        # Three conv blocks, filters increasing 32 -> 64 -> 128, each halving
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

        # Global average pooling collapses each of the 128 feature maps down
        # to a single number, giving a 128-length vector regardless of the
        # spatial size going in -- keeps the classifier head small.
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
        return x  # raw class scores (logits) -- no softmax here, the loss function applies it
