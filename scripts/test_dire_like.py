import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.dire_like.extract_maps import (
    extract_dire_like_map,
    extract_dire_like_gray_map,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = T2IImageNetWrapper(device=device)

    # Fake test image normalized in [-1, 1]
    x = torch.randn(1, 3, 256, 256).clamp(-1, 1).to(device)

    error_map = extract_dire_like_map(x, wrapper)
    gray_map = extract_dire_like_gray_map(x, wrapper)

    print("Input image:", x.shape)
    print("DIRE-like RGB map:", error_map.shape)
    print("DIRE-like gray map:", gray_map.shape)

    print("RGB map min/max:", error_map.min().item(), error_map.max().item())
    print("Gray map min/max:", gray_map.min().item(), gray_map.max().item())


if __name__ == "__main__":
    main()