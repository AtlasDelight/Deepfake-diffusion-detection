import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet backbone that returns the last convolutional feature map
    instead of final classification logits.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            self.out_channels = 512

        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            self.out_channels = 2048

        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                "Expected 'resnet18' or 'resnet50'."
            )

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ErrorGuidedSpatialRefinement(nn.Module):
    """
    Spatial refinement guided by the LaRE-like map.

    The LaRE map is resized to the feature-map resolution, then transformed
    into a spatial attention map.
    """

    def __init__(self, lare_channels: int = 4, hidden_channels: int = 32):
        super().__init__()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(lare_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, image_feature, lare_map):
        target_size = image_feature.shape[-2:]

        lare_aligned = F.interpolate(
            lare_map,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        spatial_weight = self.spatial_attention(lare_aligned)

        refined_feature = image_feature * (1.0 + spatial_weight)

        return refined_feature, spatial_weight


class ErrorGuidedChannelRefinement(nn.Module):
    """
    Channel refinement guided by the LaRE-like map.

    The LaRE map is resized and projected to the feature channel dimension.
    Global pooling and a small MLP produce channel weights.
    """

    def __init__(
        self,
        lare_channels: int,
        feature_channels: int,
        reduction: int = 16,
    ):
        super().__init__()

        hidden = max(feature_channels // reduction, 16)

        self.lare_projection = nn.Sequential(
            nn.Conv2d(lare_channels, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        self.channel_mlp = nn.Sequential(
            nn.Linear(feature_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_channels),
            nn.Sigmoid(),
        )

    def forward(self, image_feature, lare_map):
        target_size = image_feature.shape[-2:]

        lare_aligned = F.interpolate(
            lare_map,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        lare_projected = self.lare_projection(lare_aligned)

        pooled = F.adaptive_avg_pool2d(lare_projected, output_size=(1, 1))
        pooled = pooled.flatten(1)

        channel_weight = self.channel_mlp(pooled)
        channel_weight = channel_weight[:, :, None, None]

        refined_feature = image_feature * (1.0 + channel_weight)

        return refined_feature, channel_weight


class LaRE2LikeClassifier(nn.Module):
    """
    LaRE2-inspired classifier.

    Inputs:
        image:    [B, 3, H, W]
        lare_map: [B, C_lare, H_lare, W_lare]

    Pipeline:
        image -> ResNet backbone -> feature map
        LaRE map -> spatial refinement
        LaRE map -> channel refinement
        refined feature map -> classifier head
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        lare_channels: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = ResNetFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
        )

        feature_channels = self.backbone.out_channels

        self.spatial_refinement = ErrorGuidedSpatialRefinement(
            lare_channels=lare_channels,
            hidden_channels=32,
        )

        self.channel_refinement = ErrorGuidedChannelRefinement(
            lare_channels=lare_channels,
            feature_channels=feature_channels,
            reduction=16,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_channels, num_classes),
        )

    def forward(self, image, lare_map, return_attention: bool = False):
        feature = self.backbone(image)

        feature_spatial, spatial_weight = self.spatial_refinement(
            image_feature=feature,
            lare_map=lare_map,
        )

        feature_refined, channel_weight = self.channel_refinement(
            image_feature=feature_spatial,
            lare_map=lare_map,
        )

        logits = self.classifier(feature_refined)

        if return_attention:
            return {
                "logits": logits,
                "spatial_weight": spatial_weight,
                "channel_weight": channel_weight,
                "feature": feature,
                "feature_refined": feature_refined,
            }

        return logits