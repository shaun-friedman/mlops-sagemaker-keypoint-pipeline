import argparse
import json
import os
import pathlib
import tarfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

CLASS_NAMES = ["non_cat", "cat"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


class ClassifierDataset(Dataset):
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
        return image, int(row["label"])


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) formula."""
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return safe_divide(ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2, n_pos * n_neg)


def class_metrics(tp, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "support": int(tp + fn),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
    with tarfile.open(f"{MODEL_DIR}/model.tar.gz", "r:gz") as tar:
        tar.extractall(path=MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(f"{MODEL_DIR}/model.pth", map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {MODEL_DIR}/model.pth")

# ---------------------------------------------------------------------------
# Generate Predictions
# ---------------------------------------------------------------------------
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_dataset = ClassifierDataset(TEST_DIR, transform=transform)
    print(f"Test samples: {len(eval_dataset)}")

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=min(8, os.cpu_count() or 1))

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in eval_loader:
            logits = model(images.to(device))
            # argmax matches the predicted_class returned by inference.py
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_labels.append(labels.numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    cat = class_metrics(tp, fp, fn)
    non_cat = class_metrics(tn, fn, fp)

    # Layout follows SageMaker's model-quality format so the registry can display it.
    # Binary metrics treat "cat" (label 1) as the positive class.
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": safe_divide(tp + tn, len(labels)), "standard_deviation": "NaN"},
            "precision": {"value": cat["precision"], "standard_deviation": "NaN"},
            "recall": {"value": cat["recall"], "standard_deviation": "NaN"},
            "f1": {"value": cat["f1"], "standard_deviation": "NaN"},
            "auc": {"value": roc_auc(labels, probs), "standard_deviation": "NaN"},
            "true_negative_rate": {"value": safe_divide(tn, tn + fp), "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": tn, "1": fp},
                "1": {"0": fn, "1": tp},
            },
        },
        "per_class_metrics": {"non_cat": non_cat, "cat": cat},
        "test_samples": int(len(labels)),
    }
    print(json.dumps(report_dict, indent=2))

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_DIR}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
