import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from other_methods.anl_like.dataset import PairedANLFolderDataset
from other_methods.anl_like.model import ANLLikeDetector


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, noises, labels in tqdm(loader, leave=False):
        images = images.to(device)
        noises = noises.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images, noises)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--train_noise_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--val_noise_dir", required=True)

    parser.add_argument("--output_dir", default="checkpoints/anl_like")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--noise_channels", type=int, default=4)
    parser.add_argument("--attention_hidden_channels", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairedANLFolderDataset(
        image_root=args.train_image_dir,
        noise_root=args.train_noise_dir,
        image_size=args.image_size,
        expected_noise_channels=args.noise_channels,
    )

    val_dataset = PairedANLFolderDataset(
        image_root=args.val_image_dir,
        noise_root=args.val_noise_dir,
        image_size=args.image_size,
        expected_noise_channels=args.noise_channels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ANLLikeDetector(
        backbone=args.backbone,
        pretrained=args.pretrained,
        noise_channels=args.noise_channels,
        attention_hidden_channels=args.attention_hidden_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            },
            output_dir / "last.pt",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                output_dir / "best.pt",
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()