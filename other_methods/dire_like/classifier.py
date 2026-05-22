import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


class DIRELikeResNetClassifier(nn.Module):
    """
    ResNet classifier for DIRE-like reconstruction error maps.

    Input:
        DIRE-like map [B, 3, H, W]

    Output:
        logits [B, 2]
    """

    def __init__(self, backbone="resnet18", pretrained=False, num_classes=2):
        super().__init__()

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                "Expected 'resnet18' or 'resnet50'."
            )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)