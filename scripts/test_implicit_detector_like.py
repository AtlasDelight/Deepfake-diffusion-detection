import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.implicit_detector_like.extract_maps import (
    extract_implicit_detector_like_features,
    extract_implicit_detector_like_gray_map,
    extract_implicit_detector_like_upsampled_map,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = T2IImageNetWrapper(device=device)

    x = torch.randn(1, 3, 256, 256).clamp(-1, 1).to(device)

    gammas = [0.2, 0.5, 0.8]

    features = extract_implicit_detector_like_features(
        x=x,
        wrapper=wrapper,
        gammas=gammas,
        prompt="a photo",
    )

    gray_map = extract_implicit_detector_like_gray_map(
        x=x,
        wrapper=wrapper,
        gammas=gammas,
        prompt="a photo",
    )

    upsampled_map = extract_implicit_detector_like_upsampled_map(
        x=x,
        wrapper=wrapper,
        gammas=gammas,
        prompt="a photo",
    )

    print("Input image:", x.shape)
    print("Implicit detector-like features:", features.shape)
    print("Implicit detector-like gray map:", gray_map.shape)
    print("Implicit detector-like upsampled map:", upsampled_map.shape)

    print("Features min/max:", features.min().item(), features.max().item())
    print("Gray map min/max:", gray_map.min().item(), gray_map.max().item())
    print("Upsampled map min/max:", upsampled_map.min().item(), upsampled_map.max().item())


if __name__ == "__main__":
    main()