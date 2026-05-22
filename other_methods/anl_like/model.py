import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedNoiseAttention(nn.Module):
    """
    Learnable attention module guided by noise-related features.

    Input:
        noise_features: [B, C_noise, H_latent, W_latent]

    Output:
        attention: [B, 1, H_image, W_image]
    """

    def __init__(self, in_channels: int = 4, hidden_channels: int = 16):
        super().__init__()

        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, noise_features: torch.Tensor, target_size):
        attention = self.attention_net(noise_features)

        attention = F.interpolate(
            attention,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return attention


class ANLLikeDetector(nn.Module):
    """
    ANL-like detector with learned attention.

    The attention is learned during classifier training.
    It is not handcrafted from the noise magnitude.
    """

    def __init__(
        self,
        noise_channels: int = 4,
        hidden_channels: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()

        self.attention_module = LearnedNoiseAttention(
            in_channels=noise_channels,
            hidden_channels=16,
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(hidden_channels * 4, num_classes),
        )

    def forward(self, image_error: torch.Tensor, noise_features: torch.Tensor):
        """
        Args:
            image_error: [B, 3, H, W]
            noise_features: [B, C_noise, H_latent, W_latent]

        Returns:
            logits: [B, 2]
            attention: [B, 1, H, W]
            guided_error: [B, 3, H, W]
        """

        attention = self.attention_module(
            noise_features,
            target_size=image_error.shape[-2:],
        )

        guided_error = image_error * attention

        logits = self.classifier(guided_error)

        return logits, attention, guided_error