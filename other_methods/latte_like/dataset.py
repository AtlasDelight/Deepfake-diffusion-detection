from pathlib import Path

import torch
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


class PairedLATTEFolderDataset(Dataset):
    """
    Dataset for LATTE-like classification using mirrored folders.

    Expected structure:

        image_root/
            real/
                img001.jpg
            fake/
                img002.jpg

        sequence_root/
            real/
                img001.pt
            fake/
                img002.pt

    Each .pt file should contain either:
        - a tensor [K, C, H, W]
        - a dict with key "sequence"

    Each sample returns:
        image, sequence, label
    """

    def __init__(
        self,
        image_root,
        sequence_root,
        image_size: int = 224,
        expected_steps: int | None = None,
        expected_channels: int = 4,
        strict: bool = True,
    ):
        self.image_root = Path(image_root)
        self.sequence_root = Path(sequence_root)
        self.image_size = image_size
        self.expected_steps = expected_steps
        self.expected_channels = expected_channels
        self.strict = strict

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")

        if not self.sequence_root.exists():
            raise FileNotFoundError(f"LATTE sequence folder not found: {self.sequence_root}")

        self.records = []
        missing_sequences = []

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

            sequence_path = self.sequence_root / relative_path.with_suffix(".pt")

            if not sequence_path.exists():
                missing_sequences.append((image_path, sequence_path))
                continue

            self.records.append(
                {
                    "image_path": image_path,
                    "sequence_path": sequence_path,
                    "label": label,
                }
            )

        if missing_sequences and strict:
            examples = "\n".join(
                [
                    f"Image: {img} -> Missing sequence: {seq}"
                    for img, seq in missing_sequences[:10]
                ]
            )
            raise FileNotFoundError(
                f"{len(missing_sequences)} LATTE sequence files are missing.\n"
                f"Examples:\n{examples}"
            )

        if len(self.records) == 0:
            raise RuntimeError(
                f"No paired image/sequence samples found.\n"
                f"image_root={self.image_root}\n"
                f"sequence_root={self.sequence_root}"
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

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def _load_sequence(self, path: Path) -> torch.Tensor:
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            if "sequence" in obj:
                sequence = obj["sequence"]
            elif "latte_sequence" in obj:
                sequence = obj["latte_sequence"]
            else:
                raise ValueError(f"Cannot find sequence tensor in dict file: {path}")
        else:
            sequence = obj

        sequence = sequence.float()

        if sequence.ndim != 4:
            raise ValueError(
                f"Expected LATTE sequence shape [K,C,H,W], got {sequence.shape} for {path}"
            )

        k, c, h, w = sequence.shape

        if self.expected_steps is not None and k != self.expected_steps:
            raise ValueError(
                f"Expected {self.expected_steps} steps, got {k} for {path}"
            )

        if c != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} latent channels, got {c} for {path}"
            )

        return sequence

    def __getitem__(self, index):
        record = self.records[index]

        image = self._load_image(record["image_path"])
        sequence = self._load_sequence(record["sequence_path"])
        label = torch.tensor(record["label"], dtype=torch.long)

        return image, sequence, label