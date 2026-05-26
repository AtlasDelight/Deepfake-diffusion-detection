import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from other_methods.dnf_like.dataset import DNFLikeMapDataset
from other_methods.dnf_like.classifier import DNFLikeCNNClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for maps, labels in tqdm(loader, desc="Training", leave=False):
        maps = maps.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(maps)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * maps.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += maps.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for maps, labels in tqdm(loader, desc="Validation", leave=False):
        maps = maps.to(device)
        labels = labels.to(device)

        logits = model(maps)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * maps.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += maps.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(path, epoch, model, optimizer, val_acc, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a DNF-like classifier on precomputed diffusion noise features."
    )

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/dnf_like")

    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--resize_to", type=int, default=None)

    parser.add_argument(
        "--normalize",
        type=str,
        default="standard",
        choices=["none", "minmax", "standard"],
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DNFLikeMapDataset(
        root_dir=args.train_dir,
        in_channels=args.in_channels,
        resize_to=args.resize_to,
        normalize=args.normalize,
    )

    val_dataset = DNFLikeMapDataset(
        root_dir=args.val_dir,
        in_channels=args.in_channels,
        resize_to=args.resize_to,
        normalize=args.normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = DNFLikeCNNClassifier(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0

    print("=" * 80)
    print("Training DNF-like classifier")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Input channels: {args.in_channels}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Resize to: {args.resize_to}")
    print(f"Normalization: {args.normalize}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        save_checkpoint(
            path=output_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            val_acc=val_acc,
            args=args,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_checkpoint(
                path=output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_acc=val_acc,
                args=args,
            )

            print(f"Best model saved: {output_dir / 'best.pt'}")

    print("=" * 80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
