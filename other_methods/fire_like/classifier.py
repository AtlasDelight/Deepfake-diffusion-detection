import torch.nn as nn

from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


class FIRELikeResNetClassifier(nn.Module):
    """
    ResNet classifier for FIRE-like frequency reconstruction error maps.

    Input:
        FIRE-like map [B, 3, H, W]

    The expected input is:

        |F(x) - F(x_hat)|

    where:
        - x is the input image
        - x_hat is the reconstructed image
        - F is the Fourier transform

    Output:
        logits [B, 2]
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        in_channels: int = 3,
        num_classes: int = 2,
    ):
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

        if in_channels != 3:
            old_conv = self.model.conv1

            self.model.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
