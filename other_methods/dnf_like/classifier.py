import torch.nn as nn


class DNFLikeCNNClassifier(nn.Module):
    """
    CNN classifier for DNF-like noise features.

    Input:
        DNF-like feature map [B, C, H, W]

    Typical inputs:
        [B, 4, 32, 32]      -> one gamma
        [B, 12, 32, 32]     -> three gammas
        [B, 16, 32, 32]     -> four gammas

    Output:
        logits [B, 2]
    """

    def __init__(
        self,
        in_channels: int = 4,
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
        feature = self.features(x)
        logits = self.classifier(feature)
        return logits
