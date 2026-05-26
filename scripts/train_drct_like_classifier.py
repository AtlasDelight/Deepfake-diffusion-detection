import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from other_methods.drct_like.dataset import PairedDRCTFolderDataset
from other_methods.drct_like.model import DRCTLikeClassifier


def margin_contrastive_loss(embeddings, labels, margin=1.0):
    embeddings = F.normalize(embeddings, dim=1)

    distances = torch.cdist(embeddings, embeddings, p=2)

    labels = labels.view(-1, 1)
    same_class = labels.eq(labels.T)

    eye = torch.eye(
        labels.size(0),
        dtype=torch.bool,
        device=labels.device,
    )

    positive_mask = same_class & ~eye
    negative_mask = ~same_class

    if positive_mask.any():
        positive_loss = distances[positive_mask].pow(2).mean()
    else:
        positive_loss = torch.tensor(0.0, device=embeddings.device)

    if negative_mask.any():
        negative_loss = F.relu(margin - distances[negative_mask]).pow(2).mean()
    else:
        negative_loss = torch.tensor(0.0, device=embeddings.device)

    return positive_loss + negative_loss


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    lambda_contrastive=0.1,
    margin=1.0,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrastive_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, reconstructions, labels in tqdm(loader, leave=False):
        images = images.to(device)
        reconstructions = reconstructions.to(device)
        labels = labels.to(device)

        all_images = torch.cat([images, reconstructions], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)

        with torch.set_grad_enabled(is_train):
            logits, embeddings = model(all_images, return_embedding=True)

            cls_loss = criterion(logits, all_labels)
            contrastive_loss = margin_contrastive_loss(
                embeddings=embeddings,
                labels=all_labels,
                margin=margin,
            )

            loss = cls_loss + lambda_contrastive * contrastive_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * all_images.size(0)
        total_cls_loss += cls_loss.item() * all_images.size(0)
        total_contrastive_loss += contrastive_loss.item() * all_images.size(0)
        total_correct += (preds == all_labels).sum().item()
        total_samples += all_images.size(0)

    return {
        "loss": total_loss / total_samples,
        "cls_loss": total_cls_loss / total_samples,
        "contrastive_loss": total_contrastive_loss / total_samples,
        "acc": total_correct / total_samples,
    }
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--train_reconstruction_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--val_reconstruction_dir", required=True)

    parser.add_argument("--output_dir", default="checkpoints/drct_like")

    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairedDRCTFolderDataset(
        image_root=args.train_image_dir,
        reconstruction_root=args.train_reconstruction_dir,
        image_size=args.image_size,
    )

    val_dataset = PairedDRCTFolderDataset(
        image_root=args.val_image_dir,
        reconstruction_root=args.val_reconstruction_dir,
        image_size=args.image_size,
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

    model = DRCTLikeClassifier(
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            lambda_contrastive=args.lambda_contrastive,
            margin=args.margin,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            lambda_contrastive=args.lambda_contrastive,
            margin=args.margin,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train acc: {train_metrics['acc']:.4f} | "
            f"Val acc: {val_metrics['acc']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["acc"],
                "args": vars(args),
            },
            output_dir / "last.pt",
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_metrics["acc"],
                    "args": vars(args),
                },
                output_dir / "best.pt",
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()