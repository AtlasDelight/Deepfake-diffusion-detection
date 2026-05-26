import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.latte_like.extract_maps import (
    extract_latte_like_sequence,
    extract_latte_like_response_sequence,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_gammas(gammas_str: str) -> list[float]:
    return [float(g.strip()) for g in gammas_str.split(",") if g.strip()]


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )


def load_image(path: Path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


def collect_images(input_dir: Path):
    image_paths = []

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()

        if class_name not in {
            "real",
            "fake",
            "true",
            "false",
            "0_real",
            "1_fake",
            "0",
            "1",
        }:
            print(f"[WARNING] Ignored folder: {class_dir}")
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)

    return image_paths


@torch.no_grad()
def extract_sequence(
    x: torch.Tensor,
    wrapper,
    mode: str,
    gammas: list[float],
    prompt: str,
    shared_noise: bool,
) -> torch.Tensor:
    if mode == "latent":
        return extract_latte_like_sequence(
            x=x,
            wrapper=wrapper,
            gammas=gammas,
            prompt=prompt,
            shared_noise=shared_noise,
        )

    if mode == "response":
        return extract_latte_like_response_sequence(
            x=x,
            wrapper=wrapper,
            gammas=gammas,
            prompt=prompt,
            shared_noise=shared_noise,
        )

    raise ValueError(f"Unknown mode: {mode}. Expected 'latent' or 'response'.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LATTE-like latent trajectory sequences from images."
    )

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument(
        "--gammas",
        type=str,
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated gamma values, for example: '0.2,0.4,0.6,0.8'.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="latent",
        choices=["latent", "response"],
        help=(
            "latent: save reconstructed latent sequence. "
            "response: save predicted noise-response sequence."
        ),
    )

    parser.add_argument("--prompt", type=str, default="a photo")
    parser.add_argument("--shared_noise", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    gammas = parse_gammas(args.gammas)

    transform = build_transform(args.image_size)
    wrapper = T2IImageNetWrapper(device=args.device)

    image_paths = collect_images(input_dir)

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    print("=" * 80)
    print("LATTE-like sequence extraction")
    print("=" * 80)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Mode: {args.mode}")
    print(f"Gammas: {gammas}")
    print(f"Shared noise: {args.shared_noise}")
    print("=" * 80)

    for image_path in tqdm(image_paths, desc="Extracting LATTE-like sequences"):
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            x = load_image(image_path, transform).unsqueeze(0)
            x = x.to(wrapper.device)

            sequence = extract_sequence(
                x=x,
                wrapper=wrapper,
                mode=args.mode,
                gammas=gammas,
                prompt=args.prompt,
                shared_noise=args.shared_noise,
            )

            sequence = sequence.squeeze(0).detach().cpu().half()

            torch.save(
                {
                    "sequence": sequence,
                    "mode": args.mode,
                    "gammas": gammas,
                    "source_image": str(image_path),
                    "shape": tuple(sequence.shape),
                },
                output_path,
            )

        except Exception as error:
            print(f"[ERROR] Failed on image: {image_path}")
            print(error)

    print("Done.")


if __name__ == "__main__":
    main()