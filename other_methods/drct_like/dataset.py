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


class PairedDRCTFolderDataset(Dataset):
    def __init__(
        self,
        image_root,
        reconstruction_root,
        image_size=224,
        strict=True,
    ):
        self.image_root = Path(image_root)
        self.reconstruction_root = Path(reconstruction_root)

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")

        if not self.reconstruction_root.exists():
            raise FileNotFoundError(
                f"Reconstruction folder not found: {self.reconstruction_root}"
            )

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

            reconstruction_path = self.reconstruction_root / relative_path.with_suffix(".png")

            if not reconstruction_path.exists():
                missing.append((image_path, reconstruction_path))
                continue

            self.records.append((image_path, reconstruction_path, label))

        if missing and strict:
            examples = "\n".join(
                [f"{img} -> {rec}" for img, rec in missing[:10]]
            )
            raise FileNotFoundError(
                f"{len(missing)} reconstruction files are missing.\n{examples}"
            )

        if len(self.records) == 0:
            raise RuntimeError("No paired image/reconstruction samples found.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.records)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, index):
        image_path, reconstruction_path, label = self.records[index]

        image = self._load_image(image_path)
        reconstruction = self._load_image(reconstruction_path)
        label = torch.tensor(label, dtype=torch.long)

        return image, reconstruction, label