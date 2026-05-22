import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.dnf_like.extract_maps import (
    extract_dnf_like_map,
    extract_dnf_like_gray_map,
    extract_dnf_like_upsampled_map,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = T2IImageNetWrapper(device=device)

    x = torch.randn(1, 3, 256, 256).clamp(-1, 1).to(device)

    noise_map = extract_dnf_like_map(
        x=x,
        wrapper=wrapper,
        gamma=0.5,
        prompt="a photo",
    )

    gray_map = extract_dnf_like_gray_map(
        x=x,
        wrapper=wrapper,
        gamma=0.5,
        prompt="a photo",
    )

    upsampled_map = extract_dnf_like_upsampled_map(
        x=x,
        wrapper=wrapper,
        gamma=0.5,
        prompt="a photo",
    )

    print("Input image:", x.shape)
    print("DNF-like noise map:", noise_map.shape)
    print("DNF-like gray map:", gray_map.shape)
    print("DNF-like upsampled map:", upsampled_map.shape)

    print("Noise map min/max:", noise_map.min().item(), noise_map.max().item())
    print("Gray map min/max:", gray_map.min().item(), gray_map.max().item())
    print("Upsampled map min/max:", upsampled_map.min().item(), upsampled_map.max().item())


if __name__ == "__main__":
    main()