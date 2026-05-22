import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.lare_like.extract_maps import (
    extract_lare_like_map,
    extract_lare_like_gray_map,
    extract_lare_like_upsampled_map,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = T2IImageNetWrapper(device=device)

    x = torch.randn(1, 3, 256, 256).clamp(-1, 1).to(device)

    latent_map = extract_lare_like_map(x, wrapper)
    gray_map = extract_lare_like_gray_map(x, wrapper)
    upsampled_map = extract_lare_like_upsampled_map(x, wrapper)

    print("Input image:", x.shape)
    print("LaRE-like latent map:", latent_map.shape)
    print("LaRE-like gray map:", gray_map.shape)
    print("LaRE-like upsampled map:", upsampled_map.shape)

    print("Latent map min/max:", latent_map.min().item(), latent_map.max().item())
    print("Gray map min/max:", gray_map.min().item(), gray_map.max().item())
    print("Upsampled map min/max:", upsampled_map.min().item(), upsampled_map.max().item())


if __name__ == "__main__":
    main()