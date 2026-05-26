import torch
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
    ResNet backbone returning visual feature maps instead of final logits.
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
                f"Unknown backbone: {backbone}. Expected 'resnet18' or 'resnet50'."
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


class VisualTokenizer(nn.Module):
    """
    Convert a visual feature map into visual tokens.
    """

    def __init__(self, in_channels: int, d_model: int = 256):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_map):
        x = self.proj(feature_map)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens


class LatentTrajectoryTokenizer(nn.Module):
    """
    Convert LATTE-like latent sequence [B,K,C,H,W] into latent tokens.

    The temporal dimension K is preserved before tokenization.
    """

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 256,
        num_steps: int = 4,
        token_grid: int = 8,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.token_grid = token_grid

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),

            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

        self.step_embedding = nn.Parameter(
            torch.zeros(1, num_steps, 1, d_model)
        )

        nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)

    def forward(self, sequence):
        """
        Args:
            sequence: [B, K, C, H, W]

        Returns:
            latent_tokens: [B, K * token_grid * token_grid, d_model]
        """

        b, k, c, h, w = sequence.shape

        if k != self.num_steps:
            raise ValueError(
                f"Expected sequence with {self.num_steps} steps, got {k}."
            )

        x = sequence.reshape(b * k, c, h, w)
        x = self.proj(x)
        x = self.pool(x)

        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(b, k, self.token_grid * self.token_grid, -1)

        x = x + self.step_embedding[:, :k]

        latent_tokens = x.reshape(b, k * self.token_grid * self.token_grid, -1)

        return latent_tokens


class CrossAttentionBlock(nn.Module):
    """
    Latent-to-visual cross-attention block.

    Latent trajectory tokens query visual tokens.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, latent_tokens, visual_tokens):
        attn_output, _ = self.cross_attention(
            query=latent_tokens,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=False,
        )

        latent_tokens = self.norm1(latent_tokens + attn_output)

        ffn_output = self.ffn(latent_tokens)
        latent_tokens = self.norm2(latent_tokens + ffn_output)

        return latent_tokens


class LATTELikeClassifier(nn.Module):
    """
    LATTE-inspired classifier.

    Inputs:
        image:    [B, 3, H, W]
        sequence: [B, K, C, H_latent, W_latent]

    Pipeline:
        image -> ResNet -> visual tokens
        LATTE sequence -> latent trajectory tokens
        cross-attention fusion
        global aggregation
        real/fake classifier
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        latent_channels: int = 4,
        num_steps: int = 4,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        token_grid: int = 8,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.visual_backbone = ResNetFeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
        )

        self.visual_tokenizer = VisualTokenizer(
            in_channels=self.visual_backbone.out_channels,
            d_model=d_model,
        )

        self.latent_tokenizer = LatentTrajectoryTokenizer(
            in_channels=latent_channels,
            d_model=d_model,
            num_steps=num_steps,
            token_grid=token_grid,
        )

        self.fusion_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, image, sequence, return_features: bool = False):
        visual_feature_map = self.visual_backbone(image)
        visual_tokens = self.visual_tokenizer(visual_feature_map)

        latent_tokens = self.latent_tokenizer(sequence)

        for layer in self.fusion_layers:
            latent_tokens = layer(
                latent_tokens=latent_tokens,
                visual_tokens=visual_tokens,
            )

        latent_global = latent_tokens.mean(dim=1)
        visual_global = visual_tokens.mean(dim=1)

        fused_global = torch.cat([latent_global, visual_global], dim=1)

        logits = self.classifier(fused_global)

        if return_features:
            return {
                "logits": logits,
                "latent_tokens": latent_tokens,
                "visual_tokens": visual_tokens,
                "fused_global": fused_global,
            }

        return logits