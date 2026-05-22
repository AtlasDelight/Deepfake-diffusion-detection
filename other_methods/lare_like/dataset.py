from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def infer_label_from_folder(folder_name: str) -> int:
    """
    Infer binary label from folder name.

    0 -> real
    1 -> fake
    """
    name = folder_name.lower()

    if name in {"real", "true", "0", "0_real"}:
        return 0

    if name in {"fake", "false", "1", "1_fake"}:
        return 1

    raise ValueError(f"Unknown class folder name: {folder_name}")


class PairedLaREFolderDataset(Dataset):
    """
    Dataset for LaRE2-like classification using mirrored folders.

    Expected structure:

        image_root/
            real/
                img001.png
                img002.png
            fake/
                img003.png
                img004.png

        map_root/
            real/
                img001.pt
                img002.pt
            fake/
                img003.pt
                img004.pt

    Each sample returns:
        image, lare_map, label
    """

    def __init__(
        self,
        image_root,
        map_root,
        image_size: int = 224,
        lare_channels: int = 4,
        resize_lare_to=None,
        strict: bool = True,
    ):
        self.image_root = Path(image_root)
        self.map_root = Path(map_root)
        self.image_size = image_size
        self.lare_channels = lare_channels
        self.resize_lare_to = resize_lare_to
        self.strict = strict

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")

        if not self.map_root.exists():
            raise FileNotFoundError(f"LaRE map folder not found: {self.map_root}")

        self.records = []
        missing_maps = []

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

            map_relative_path = relative_path.with_suffix(".pt")
            map_path = self.map_root / map_relative_path

            if not map_path.exists():
                missing_maps.append((image_path, map_path))
                continue

            self.records.append(
                {
                    "image_path": image_path,
                    "map_path": map_path,
                    "label": label,
                }
            )

        if missing_maps and strict:
            examples = "\n".join(
                [
                    f"Image: {img} -> Missing map: {mp}"
                    for img, mp in missing_maps[:10]
                ]
            )
            raise FileNotFoundError(
                f"{len(missing_maps)} LaRE maps are missing.\n"
                f"Examples:\n{examples}"
            )

        if len(self.records) == 0:
            raise RuntimeError(
                f"No paired image/map samples found.\n"
                f"image_root={self.image_root}\n"
                f"map_root={self.map_root}"
            )

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

    def _load_image(self, path: Path):
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def _load_lare_map(self, path: Path):
        x = torch.load(path, map_location="cpu")

        if isinstance(x, dict):
            if "map" in x:
                x = x["map"]
            elif "error_map" in x:
                x = x["error_map"]
            elif "latent_map" in x:
                x = x["latent_map"]
            else:
                raise ValueError(f"Cannot find LaRE tensor in dict file: {path}")

        x = x.float()

        if x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim != 3:
            raise ValueError(f"Expected LaRE map [C,H,W], got {x.shape} for {path}")

        if x.shape[0] == 1 and self.lare_channels == 4:
            x = x.repeat(4, 1, 1)

        if x.shape[0] == 4 and self.lare_channels == 1:
            x = x.mean(dim=0, keepdim=True)

        if x.shape[0] != self.lare_channels:
            raise ValueError(
                f"Expected {self.lare_channels} channels, "
                f"got {x.shape[0]} for {path}"
            )

        if self.resize_lare_to is not None:
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.resize_lare_to, self.resize_lare_to),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return x

    def __getitem__(self, index):
        record = self.records[index]

        image = self._load_image(record["image_path"])
        lare_map = self._load_lare_map(record["map_path"])
        label = torch.tensor(record["label"], dtype=torch.long)

        return image, lare_map, label