import os
import argparse
import json
import pathlib
import tarfile

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision import models
from PIL import Image
from torch.utils.data import DataLoader, Dataset

KP_NAMES = [
    "left_eye", "right_eye", "mouth",
    "left_ear_1", "left_ear_2", "left_ear_3",
    "right_ear_1", "right_ear_2", "right_ear_3",
]

KP_COLS_NORM = [f"{kp}_{axis}_norm" for kp in KP_NAMES for axis in ("x", "y")]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TARGET_W, TARGET_H = 224, 224
MODEL_DIR = "/opt/ml/processing/model"
TRAIN_DIR = "/opt/ml/processing/train"
BEST_MODEL_PATH = "model_best.pth"


def keypoint_errors(pred_norm, df):
    """NME per image and keypoint for normalized predictions of shape (N, 18).

    Error per keypoint in original-image pixels, divided by the ground-truth
    inter-ocular distance (left_eye to right_eye), so it is comparable across
    image sizes and how large the cat's face is in the frame. Images whose two
    eye annotations coincide have no usable scale and are excluded."""
    pred_norm = np.asarray(pred_norm, dtype="float64").reshape(-1, len(KP_NAMES), 2)
    gt_norm = df[KP_COLS_NORM].to_numpy(dtype="float64").reshape(-1, len(KP_NAMES), 2)
    scale = df[["width", "height"]].to_numpy(dtype="float64")[:, None, :]

    pred_px = pred_norm * scale
    gt_px = gt_norm * scale
    inter_ocular = np.linalg.norm(gt_px[:, 0] - gt_px[:, 1], axis=1)

    valid = inter_ocular > 0
    nme = np.linalg.norm(pred_px - gt_px, axis=2)[valid] / inter_ocular[valid, None]
    return nme, valid


def nme_summary(nme_per_image):
    return {
        "nme_mean": float(nme_per_image.mean()),
        "nme_median": float(np.median(nme_per_image)),
        "nme_p95": float(np.percentile(nme_per_image, 95)),
        "failure_rate_at_0_1": float((nme_per_image > 0.1).mean()),
    }


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

if __name__ == "__main__":
    test_dir = "/opt/ml/processing/test"    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
    model_tar_path = f"{MODEL_DIR}/model.tar.gz"

    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=MODEL_DIR)

    model = KeypointModel(num_keypoints=9, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/{BEST_MODEL_PATH}", map_location=device)
    )
    model.eval()
    print(f"Model loaded from {MODEL_DIR}/{BEST_MODEL_PATH}")

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------   
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ---------------------------------------------------------------------------
# Annotations Ground Truth
# ---------------------------------------------------------------------------   

    eval_dataset = KeypointDataset(test_dir, transform=transform)

    print(f"Test samples: {len(eval_dataset)}")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=device.type == "cuda",
    )

# ---------------------------------------------------------------------------
# Generate Predictions
# --------------------------------------------------------------------------- 
    model.eval()
    criterion = nn.MSELoss()
    eval_loss = 0.0
    batch_predictions = []
    with torch.no_grad():
        for images, targets in eval_loader:
            images  = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)
            eval_loss += loss.item() * images.size(0)
            batch_predictions.append(predictions.cpu().numpy())

    eval_loss /= len(eval_loader.dataset)
    print(f"Test Loss: {eval_loss:.6f}")

# ---------------------------------------------------------------------------
# Normalized Mean Error (NME)
# ---------------------------------------------------------------------------
    nme, valid = keypoint_errors(np.concatenate(batch_predictions), eval_dataset.df)

# ---------------------------------------------------------------------------
# Benchmark: mean shape
# ---------------------------------------------------------------------------
    # Predict the average training keypoint positions for every image. The model
    # has to beat this to show it locates the face rather than learning the average layout.
    train_annotations = pd.read_parquet(os.path.join(TRAIN_DIR, "annotations.parquet"))
    mean_shape = train_annotations[KP_COLS_NORM].to_numpy(dtype="float64").mean(axis=0)
    mean_shape_pred = np.tile(mean_shape, (len(eval_dataset.df), 1))
    mean_shape_nme, _ = keypoint_errors(mean_shape_pred, eval_dataset.df)
    gt_norm = eval_dataset.df[KP_COLS_NORM].to_numpy(dtype="float64")

    model_nme = nme_summary(nme.mean(axis=1))
    mean_shape_nme_summary = nme_summary(mean_shape_nme.mean(axis=1))

    report_dict = {
        "regression_metrics": {
            "mse": {"value": eval_loss},
            "rmse": {"value": float(np.sqrt(eval_loss))},
            **{name: {"value": value} for name, value in model_nme.items()},
            # The pipeline's quality gate: the model's mean NME as a fraction of the
            # mean-shape benchmark's. 1.0 would mean no better than the average layout.
            "nme_ratio_to_mean_shape": {
                "value": model_nme["nme_mean"] / mean_shape_nme_summary["nme_mean"]
            },
        },
        "per_keypoint_nme": {
            name: float(value) for name, value in zip(KP_NAMES, nme.mean(axis=0))
        },
        "test_samples": int(valid.sum()),
        "excluded_samples": int((~valid).sum()),
        "benchmarks": {
            "mean_shape": {
                "mse": float(((mean_shape_pred - gt_norm) ** 2).mean()),
                **mean_shape_nme_summary,
                "per_keypoint_nme": {
                    name: float(value) for name, value in zip(KP_NAMES, mean_shape_nme.mean(axis=0))
                },
            },
        },
    }
    print(json.dumps(report_dict, indent=2))

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

