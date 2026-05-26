from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


MAP_EXTENSIONS = {".pt", ".pth"}


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


class DNFLikeMapDataset(Dataset):
    """
    Dataset for precomputed DNF-like noise features.

    Expected structure:

        root/
            real/
                img001.pt
                img002.pt
            fake/
                img003.pt
                img004.pt

    Each .pt file should contain a tensor such as:

        [4, 32, 32]

    or, for a multi-gamma DNF-like representation:

        [K * 4, 32, 32]

    Each sample returns:
        dnf_map, label
    """

    def __init__(
        self,
        root_dir,
        in_channels: int = 4,
        resize_to=None,
        normalize: str = "standard",
    ):
        self.root_dir = Path(root_dir)
        self.in_channels = in_channels
        self.resize_to = resize_to
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
                f"No DNF-like maps found in {self.root_dir}. "
                "Expected real/ and fake/ folders containing .pt files."
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

    def _fix_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim != 3:
            raise ValueError(f"Expected DNF map shape [C,H,W], got {x.shape}")

        if x.shape[0] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"got {x.shape[0]}. "
                "If you use multiple gammas, set --in_channels accordingly."
            )

        if self.resize_to is not None:
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return x

    def _load_map(self, path: Path) -> torch.Tensor:
        x = torch.load(path, map_location="cpu")

        if isinstance(x, dict):
            if "dnf_map" in x:
                x = x["dnf_map"]
            elif "noise_feature" in x:
                x = x["noise_feature"]
            elif "noise_error" in x:
                x = x["noise_error"]
            elif "map" in x:
                x = x["map"]
            elif "feature" in x:
                x = x["feature"]
            else:
                raise ValueError(f"Cannot find DNF tensor in dict file: {path}")

        x = x.float()
        x = self._fix_shape(x)
        x = self._normalize_tensor(x)

        return x

    def __getitem__(self, index):
        path, label = self.records[index]

        x = self._load_map(path)
        y = torch.tensor(label, dtype=torch.long)

        return x, y