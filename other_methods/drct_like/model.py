import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


class DRCTLikeClassifier(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        pretrained=False,
        embedding_dim=None,
        num_classes=2,
    ):
        super().__init__()

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)

        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)

        else:
            raise ValueError("backbone must be 'resnet18' or 'resnet50'")

        feature_dim = model.fc.in_features
        model.fc = nn.Identity()

        self.backbone = model

        if embedding_dim is None:
            embedding_dim = feature_dim

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        features = self.backbone(x)
        embedding = self.projector(features)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, F.normalize(embedding, dim=1)

        return logits