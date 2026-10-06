import argparse
import os
import shutil

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClassifierDataset(Dataset):
    """Images written by preprocess.py, already resized to 224x224, with 0/1 labels."""

    def __init__(self, channel_dir: str, transform=None):
        self.channel_dir = channel_dir
        self.transform = transform
        self.df = pd.read_parquet(os.path.join(channel_dir, "labels.parquet"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.channel_dir, row["image"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(int(row["label"]), dtype=torch.long)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(num_classes: int = 2) -> nn.Module:
    """ImageNet ResNet-18 with a frozen backbone and a new classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ImageNet normalization, matching src/classification/inference.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_one_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train(args):
    # SageMaker injects data channel paths as environment variables
    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    val_dir   = os.environ["SM_CHANNEL_VALIDATION"]
    model_dir = os.environ["SM_MODEL_DIR"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ClassifierDataset(train_dir, transform=transform)
    val_dataset   = ClassifierDataset(val_dir,   transform=transform)
    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")
    print(f"Train label counts: {train_dataset.df['label'].value_counts().to_dict()}")

    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model     = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    # Only the new head is trainable
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)
        scheduler.step(val_loss)

        # Parsed by the estimator's metric_definitions
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # inference.py loads model.pth, so the best checkpoint is the one served
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size",    type=int,   default=64)
    args = parser.parse_args()

    train(args)

    # Package the serving script so the registered model deploys without repacking
    code_dir = os.path.join(os.environ["SM_MODEL_DIR"], "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py"), code_dir)
