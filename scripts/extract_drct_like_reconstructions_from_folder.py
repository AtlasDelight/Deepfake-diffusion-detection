import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.light_model.t2i_imagenet_wrapper import T2IImageNetWrapper
from other_methods.drct_like.extract_maps import extract_drct_like_reconstruction


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def collect_images(input_dir):
    input_dir = Path(input_dir)
    images = []

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        if class_dir.name.lower() not in {"real", "fake", "true", "false", "0", "1"}:
            continue

        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)

    return images


def save_tensor_as_image(x, output_path):
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    image = TF.to_pil_image(x)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(args.image_size)
    wrapper = T2IImageNetWrapper(device=args.device)

    image_paths = collect_images(input_dir)

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    for image_path in tqdm(image_paths, desc="Extracting DRCT reconstructions"):
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            image = Image.open(image_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(wrapper.device)

            x_hat = extract_drct_like_reconstruction(
                x=x,
                wrapper=wrapper,
            )

            save_tensor_as_image(x_hat.squeeze(0), output_path)

        except Exception as error:
            print(f"[ERROR] {image_path}: {error}")

    print("Done.")


if __name__ == "__main__":
    main()