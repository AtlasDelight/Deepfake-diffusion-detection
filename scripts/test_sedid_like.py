import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.sedid_like.extract_maps import (
    extract_sedid_like_map,
    extract_sedid_like_gray_map,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = T2IImageNetWrapper(device=device)

    x = torch.randn(1, 3, 256, 256).clamp(-1, 1).to(device)

    sedid_map = extract_sedid_like_map(
        x=x,
        wrapper=wrapper,
        gamma_source=0.5,
        gamma_target=0.8,
        prompt="a photo",
    )

    gray_map = extract_sedid_like_gray_map(
        x=x,
        wrapper=wrapper,
        gamma_source=0.5,
        gamma_target=0.8,
        prompt="a photo",
    )

    print("Input image:", x.shape)
    print("SeDID-like image map:", sedid_map.shape)
    print("SeDID-like gray map:", gray_map.shape)

    print("Image map min/max:", sedid_map.min().item(), sedid_map.max().item())
    print("Gray map min/max:", gray_map.min().item(), gray_map.max().item())


if __name__ == "__main__":
    main()