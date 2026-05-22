from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MAP_EXTENSIONS = {".pt", ".pth", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def infer_label_from_folder(folder_name: str) -> int:
    name = folder_name.lower()

    if name in {"real", "true", "0", "0_real"}:
        return 0

    if name in {"fake", "false", "1", "1_fake"}:
        return 1

    raise ValueError(f"Unknown class folder name: {folder_name}")


class DIRELikeMapDataset(Dataset):
    """
    Dataset for precomputed DIRE-like error maps.

    Expected structure:

        root/
            real/
                map_001.pt
                map_002.pt
            fake/
                map_003.pt
                map_004.pt

    or:

        root/
            0_real/
            1_fake/

    Each sample is a reconstruction error map and a binary label:
        0 -> real
        1 -> fake
    """

    def __init__(self, root_dir, image_size=224):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.records = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            try:
                label = infer_label_from_folder(class_dir.name)
            except ValueError:
                continue

            for path in class_dir.rglob("*"):
                if path.suffix.lower() in MAP_EXTENSIONS:
                    self.records.append((path, label))

        if len(self.records) == 0:
            raise RuntimeError(
                f"No DIRE-like maps found in {self.root_dir}. "
                "Expected folders such as real/ and fake/ containing .pt or image files."
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.records)

    def _load_pt_map(self, path: Path) -> torch.Tensor:
        x = torch.load(path, map_location="cpu")

        if isinstance(x, dict):
            if "map" in x:
                x = x["map"]
            elif "error_map" in x:
                x = x["error_map"]
            else:
                raise ValueError(f"Cannot find map tensor in dict file: {path}")

        x = x.float()

        if x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim != 3:
            raise ValueError(f"Expected map shape [C,H,W], got {x.shape} for {path}")

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        if x.shape[0] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[0]} for {path}")

        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return x

    def _load_image_map(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def __getitem__(self, index):
        path, label = self.records[index]

        if path.suffix.lower() in {".pt", ".pth"}:
            x = self._load_pt_map(path)
        else:
            x = self._load_image_map(path)

        y = torch.tensor(label, dtype=torch.long)

        return x, y