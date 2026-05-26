import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=False):
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
            raise ValueError("backbone must be 'resnet18' or 'resnet50'")

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


class LearnedNoiseAttention(nn.Module):
    def __init__(self, noise_channels=4, hidden_channels=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(noise_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, predicted_noise, target_size):
        attention = self.net(predicted_noise)

        attention = F.interpolate(
            attention,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return attention


class ANLLikeDetector(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        pretrained=False,
        noise_channels=4,
        attention_hidden_channels=32,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        self.backbone = ResNetFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
        )

        self.attention = LearnedNoiseAttention(
            noise_channels=noise_channels,
            hidden_channels=attention_hidden_channels,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_channels, num_classes),
        )

    def forward(self, image, predicted_noise, return_attention=False):
        feature_map = self.backbone(image)

        attention_map = self.attention(
            predicted_noise=predicted_noise,
            target_size=feature_map.shape[-2:],
        )

        refined_feature = feature_map * (1.0 + attention_map)

        logits = self.classifier(refined_feature)

        if return_attention:
            return {
                "logits": logits,
                "attention": attention_map,
                "feature_map": feature_map,
                "refined_feature": refined_feature,
            }

        return logits