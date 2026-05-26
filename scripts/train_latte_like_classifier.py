import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from other_methods.latte_like.dataset import PairedLATTEFolderDataset
from other_methods.latte_like.classifier import LATTELikeClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, sequences, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images, sequences)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, sequences, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(images, sequences)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

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
        description="Train a LATTE-like classifier with latent trajectory and visual fusion."
    )

    parser.add_argument("--train_image_dir", type=str, required=True)
    parser.add_argument("--train_sequence_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--val_sequence_dir", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="checkpoints/latte_like")

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
    )
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=4)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--token_grid", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairedLATTEFolderDataset(
        image_root=args.train_image_dir,
        sequence_root=args.train_sequence_dir,
        image_size=args.image_size,
        expected_steps=args.num_steps,
        expected_channels=args.latent_channels,
        strict=True,
    )

    val_dataset = PairedLATTEFolderDataset(
        image_root=args.val_image_dir,
        sequence_root=args.val_sequence_dir,
        image_size=args.image_size,
        expected_steps=args.num_steps,
        expected_channels=args.latent_channels,
        strict=True,
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

    model = LATTELikeClassifier(
        backbone=args.backbone,
        pretrained=args.pretrained,
        latent_channels=args.latent_channels,
        num_steps=args.num_steps,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        token_grid=args.token_grid,
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
    print("Training LATTE-like classifier")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Pretrained backbone: {args.pretrained}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image size: {args.image_size}")
    print(f"Latent channels: {args.latent_channels}")
    print(f"Number of steps: {args.num_steps}")
    print(f"d_model: {args.d_model}")
    print(f"Attention heads: {args.num_heads}")
    print(f"Fusion layers: {args.num_layers}")
    print(f"Token grid: {args.token_grid}")
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