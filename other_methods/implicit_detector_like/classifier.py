import torch.nn as nn


class ImplicitDetectorCNNClassifier(nn.Module):
    """
    CNN classifier for implicit detector-like model response features.

    Input:
        [B, C, H, W]

    Typical input:
        [B, 36, 32, 32]

    Output:
        logits [B, 2]
    """

    def __init__(
        self,
        in_channels: int = 36,
        hidden_channels: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits