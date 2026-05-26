from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def infer_label_from_folder(folder_name: str) -> int:
    name = folder_name.lower()

    if name in {"real", "true", "0", "0_real"}:
        return 0

    if name in {"fake", "false", "1", "1_fake"}:
        return 1

    raise ValueError(f"Unknown class folder name: {folder_name}")


class PairedANLFolderDataset(Dataset):
    def __init__(
        self,
        image_root,
        noise_root,
        image_size=224,
        expected_noise_channels=4,
        normalize_noise="standard",
        strict=True,
    ):
        self.image_root = Path(image_root)
        self.noise_root = Path(noise_root)
        self.expected_noise_channels = expected_noise_channels
        self.normalize_noise = normalize_noise

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")

        if not self.noise_root.exists():
            raise FileNotFoundError(f"Noise folder not found: {self.noise_root}")

        self.records = []
        missing = []

        for image_path in sorted(self.image_root.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(self.image_root)

            if len(relative_path.parts) < 2:
                continue

            class_name = relative_path.parts[0]

            try:
                label = infer_label_from_folder(class_name)
            except ValueError:
                continue

            noise_path = self.noise_root / relative_path.with_suffix(".pt")

            if not noise_path.exists():
                missing.append((image_path, noise_path))
                continue

            self.records.append((image_path, noise_path, label))

        if missing and strict:
            examples = "\n".join(
                [f"{img} -> {noise}" for img, noise in missing[:10]]
            )
            raise FileNotFoundError(
                f"{len(missing)} noise files are missing.\n{examples}"
            )

        if len(self.records) == 0:
            raise RuntimeError("No paired image/noise samples found.")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.records)

    def _normalize_noise(self, x):
        if self.normalize_noise == "none":
            return x

        if self.normalize_noise == "minmax":
            x_min = x.amin(dim=(-2, -1), keepdim=True)
            x_max = x.amax(dim=(-2, -1), keepdim=True)
            return (x - x_min) / (x_max - x_min + 1e-8)

        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-8)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def _load_noise(self, path):
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            if "predicted_noise" in obj:
                noise = obj["predicted_noise"]
            elif "noise_response" in obj:
                noise = obj["noise_response"]
            elif "noise" in obj:
                noise = obj["noise"]
            else:
                raise ValueError(f"Cannot find noise tensor in {path}")
        else:
            noise = obj

        noise = noise.float()

        if noise.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got {noise.shape}")

        if noise.shape[0] != self.expected_noise_channels:
            raise ValueError(
                f"Expected {self.expected_noise_channels} channels, got {noise.shape[0]}"
            )

        return self._normalize_noise(noise)

    def __getitem__(self, index):
        image_path, noise_path, label = self.records[index]

        image = self._load_image(image_path)
        predicted_noise = self._load_noise(noise_path)
        label = torch.tensor(label, dtype=torch.long)

        return image, predicted_noise, label