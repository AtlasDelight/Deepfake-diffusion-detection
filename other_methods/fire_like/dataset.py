from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MAP_EXTENSIONS = {".pt", ".pth", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


class FIRELikeMapDataset(Dataset):
    """
    Dataset for precomputed FIRE-like frequency reconstruction error maps.

    Expected structure:

        root/
            real/
                img001.pt
                img002.pt
            fake/
                img003.pt
                img004.pt

    Each .pt file should ideally contain a tensor:

        [3, H, W]

    corresponding to:

        |F(x) - F(x_hat)|

    where F is the Fourier transform and x_hat is the reconstructed image.

    Each sample returns:
        fire_map, label
    """

    def __init__(
        self,
        root_dir,
        image_size: int = 224,
        in_channels: int = 3,
        normalize: str = "minmax",
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.in_channels = in_channels
        self.normalize = normalize

        if self.normalize not in {"none", "minmax", "standard"}:
            raise ValueError("normalize must be one of: none, minmax, standard")

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        self.records = []

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
                f"No FIRE-like maps found in {self.root_dir}. "
                "Expected real/ and fake/ folders containing .pt or image files."
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.records)

    def _normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return x

        if self.normalize == "minmax":
            x_min = x.amin(dim=(-2, -1), keepdim=True)
            x_max = x.amax(dim=(-2, -1), keepdim=True)
            return (x - x_min) / (x_max - x_min + 1e-8)

        if self.normalize == "standard":
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True)
            return (x - mean) / (std + 1e-8)

        return x

    def _fix_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert the loaded map to the expected number of channels.
        """

        if x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim != 3:
            raise ValueError(f"Expected map shape [C,H,W], got {x.shape}")

        c = x.shape[0]

        if c == self.in_channels:
            return x

        if c == 1 and self.in_channels == 3:
            return x.repeat(3, 1, 1)

        if c == 3 and self.in_channels == 1:
            return x.mean(dim=0, keepdim=True)

        if c == 6 and self.in_channels == 3:
            # If a previous FIRE-like version saved two RGB maps concatenated,
            # reduce them to one RGB-like map by averaging both parts.
            first = x[:3]
            second = x[3:6]
            return 0.5 * (first + second)

        raise ValueError(
            f"Cannot convert map with {c} channels to expected {self.in_channels} channels."
        )

    def _resize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return x

    def _load_pt_map(self, path: Path) -> torch.Tensor:
        x = torch.load(path, map_location="cpu")

        if isinstance(x, dict):
            if "fire_map" in x:
                x = x["fire_map"]
            elif "map" in x:
                x = x["map"]
            elif "frequency_error" in x:
                x = x["frequency_error"]
            elif "freq_error" in x:
                x = x["freq_error"]
            elif "error_map" in x:
                x = x["error_map"]
            else:
                raise ValueError(f"Cannot find FIRE tensor in dict file: {path}")

        x = x.float()
        x = self._fix_channels(x)
        x = self._resize_tensor(x)
        x = self._normalize_tensor(x)

        return x

    def _load_image_map(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        x = self.image_transform(image)

        x = self._fix_channels(x)
        x = self._normalize_tensor(x)

        return x

    def __getitem__(self, index):
        path, label = self.records[index]

        if path.suffix.lower() in {".pt", ".pth"}:
            x = self._load_pt_map(path)
        else:
            x = self._load_image_map(path)

        y = torch.tensor(label, dtype=torch.long)

        return x, y