import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.fire_like.extract_maps import (
    extract_fire_like_map,
    extract_fire_like_gray_map,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = T2IImageNetWrapper(device=device)

    x = torch.randn(1, 3, 256, 256).clamp(-1, 1).to(device)

    fire_map_all = extract_fire_like_map(
        x=x,
        wrapper=wrapper,
        band="all",
    )

    fire_map_mid = extract_fire_like_map(
        x=x,
        wrapper=wrapper,
        band="mid",
    )

    gray_map = extract_fire_like_gray_map(
        x=x,
        wrapper=wrapper,
        band="mid",
    )

    print("Input image:", x.shape)
    print("FIRE-like full map:", fire_map_all.shape)
    print("FIRE-like mid-frequency map:", fire_map_mid.shape)
    print("FIRE-like gray map:", gray_map.shape)

    print("Full map min/max:", fire_map_all.min().item(), fire_map_all.max().item())
    print("Mid map min/max:", fire_map_mid.min().item(), fire_map_mid.max().item())
    print("Gray map min/max:", gray_map.min().item(), gray_map.max().item())


if __name__ == "__main__":
    main()