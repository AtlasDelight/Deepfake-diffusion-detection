import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.anl_like.features import extract_anl_like_features
from other_methods.anl_like.model import ANLLikeDetector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = T2IImageNetWrapper(device=device)

    x = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(device)

    features = extract_anl_like_features(
        x=x,
        wrapper=wrapper,
        gamma=0.5,
        prompt="a photo",
    )

    image_error = features["image_error"]
    noise_features = features["noise_response"]

    model = ANLLikeDetector(noise_channels=noise_features.shape[1]).to(device)

    logits, attention, guided_error = model(
        image_error=image_error,
        noise_features=noise_features,
    )

    print("image_error:", image_error.shape)
    print("noise_features:", noise_features.shape)
    print("attention:", attention.shape)
    print("guided_error:", guided_error.shape)
    print("logits:", logits.shape)


if __name__ == "__main__":
    main()