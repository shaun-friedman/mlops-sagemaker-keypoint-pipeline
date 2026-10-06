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

KP_COLS_NORM = [
    "left_eye_x_norm",
    "left_eye_y_norm",
    "right_eye_x_norm",
    "right_eye_y_norm",
    "mouth_x_norm",
    "mouth_y_norm",
    "left_ear_1_x_norm",
    "left_ear_1_y_norm",
    "left_ear_2_x_norm",
    "left_ear_2_y_norm",
    "left_ear_3_x_norm",
    "left_ear_3_y_norm",
    "right_ear_1_x_norm",
    "right_ear_1_y_norm",
    "right_ear_2_x_norm",
    "right_ear_2_y_norm",
    "right_ear_3_x_norm",
    "right_ear_3_y_norm",
]


class KeypointDataset(Dataset):
    def __init__(self, channel_dir: str, transform=None):
        self.channel_dir = channel_dir
        self.transform = transform
        annotations_path = os.path.join(channel_dir, "annotations.parquet")
        self.df = pd.read_parquet(annotations_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Reconstruct full path from channel_dir + filename
        image_path = os.path.join(self.channel_dir, row["image"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        keypoints = torch.tensor(row[KP_COLS_NORM].values.astype("float32"))
        return image, keypoints
    
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KeypointModel(nn.Module):
    """
    Keypoint regression model for 224x224 input images.
    Predicts 9 keypoints (18 values = 9 x (x, y) coordinates).
    """

    def __init__(self, num_keypoints: int = 9, pretrained: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        # Remove final classification head -> output: (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Regression head -> (B, 18)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_keypoints * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ImageNet normalization — standard when using a pretrained ResNet backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # SageMaker injects data channel paths as environment variables
    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    val_dir   = os.environ["SM_CHANNEL_VALIDATION"]
    model_dir = os.environ["SM_MODEL_DIR"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = KeypointDataset(train_dir, transform=train_transforms)
    val_dataset   = KeypointDataset(val_dir,   transform=val_transforms)

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    model     = KeypointModel(num_keypoints=9, pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):

        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images  = images.to(device)   # (B, 3, 224, 224)
            targets = targets.to(device)  # (B, 18)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images  = images.to(device)
                targets = targets.to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "model_best.pth"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.6f})")

    # Save final model — SageMaker uploads SM_MODEL_DIR to S3
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size",    type=int,   default=32)
    args = parser.parse_args()

    train(args)

    # Package the serving script so the registered model deploys without repacking
    code_dir = os.path.join(os.environ["SM_MODEL_DIR"], "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py"), code_dir)