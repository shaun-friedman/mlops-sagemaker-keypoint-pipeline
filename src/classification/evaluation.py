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
TRAIN_DIR = "/opt/ml/processing/train"
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


def summary_metrics(labels, preds, probs) -> dict:
    """Headline binary metrics with cat (label 1) as the positive class."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    cat = class_metrics(tp, fp, fn)
    return {
        "accuracy": safe_divide(tp + tn, len(labels)),
        "precision": cat["precision"],
        "recall": cat["recall"],
        "f1": cat["f1"],
        "auc": roc_auc(labels, probs),
        "true_negative_rate": safe_divide(tn, tn + fp),
    }

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# Statistics of the original image, recorded by preprocess.py. Nothing here describes
# what is in the image, so a model that scores well on these alone is using a shortcut
# (resolution, framing, compression) rather than recognizing cats.
BENCHMARK_FEATURES = ["width", "height", "aspect_ratio", "brightness", "contrast", "file_size"]
LOG_FEATURES = {"width", "height", "file_size"}


def benchmark_features(df: pd.DataFrame) -> np.ndarray:
    columns = [np.log(df[c]) if c in LOG_FEATURES else df[c] for c in BENCHMARK_FEATURES]
    return np.column_stack(columns).astype("float64")


def fit_logistic_regression(x, y, l2=1.0, iterations=50):
    """Logistic regression by Newton's method with a small L2 penalty. x includes a bias column."""
    weights = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-x @ weights))
        gradient = x.T @ (p - y) + l2 * weights
        hessian = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(len(weights))
        weights -= np.linalg.solve(hessian, gradient)
    return weights


def image_statistics_benchmark(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit on the training split's image statistics; return test predictions and cat probabilities."""
    x_train, x_test = benchmark_features(train_df), benchmark_features(test_df)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-12
    add_bias = lambda x: np.column_stack([np.ones(len(x)), (x - mean) / std])
    weights = fit_logistic_regression(add_bias(x_train), train_df["label"].to_numpy(dtype="float64"))
    probs = 1 / (1 + np.exp(-add_bias(x_test) @ weights))
    return (probs >= 0.5).astype(int), probs


def per_source_metrics(labels, preds, sources):
    """Accuracy per image source, with cat recall and non-cat specificity where the
    source has that class. A model that learned the source rather than the content
    does well on some sources and poorly on others."""
    results = {}
    for source in sorted(set(sources)):
        mask = sources == source
        source_labels, source_preds = labels[mask], preds[mask]
        entry = {"samples": int(mask.sum()), "accuracy": float((source_labels == source_preds).mean())}
        if (source_labels == 1).any():
            entry["cat_recall"] = float((source_preds[source_labels == 1] == 1).mean())
        if (source_labels == 0).any():
            entry["non_cat_specificity"] = float((source_preds[source_labels == 0] == 0).mean())
        results[source] = entry
    return results


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
    # DataLoader keeps dataset order (shuffle=False), so rows line up with labels.parquet
    sources = eval_dataset.df["source"].to_numpy() if "source" in eval_dataset.df else np.full(len(labels), "unknown")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    cat = class_metrics(tp, fp, fn)
    non_cat = class_metrics(tn, fn, fp)

# ---------------------------------------------------------------------------
# Benchmarks: what the model has to beat
# ---------------------------------------------------------------------------
    train_df = pd.read_parquet(os.path.join(TRAIN_DIR, "labels.parquet"))
    test_df = eval_dataset.df

    # Always predict the class that is more common in training
    majority = int(train_df["label"].mean() >= 0.5)
    majority_preds = np.full(len(labels), majority)
    majority_probs = np.full(len(labels), train_df["label"].mean())

    stats_preds, stats_probs = image_statistics_benchmark(train_df, test_df)

    benchmarks = {
        "majority_class": {
            "predicts": CLASS_NAMES[majority],
            **summary_metrics(labels, majority_preds, majority_probs),
        },
        "image_statistics": {
            "features": BENCHMARK_FEATURES,
            **summary_metrics(labels, stats_preds, stats_probs),
            "per_source_metrics": per_source_metrics(labels, stats_preds, sources),
        },
    }

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
        "per_source_metrics": per_source_metrics(labels, preds, sources),
        "test_samples": int(len(labels)),
        "benchmarks": benchmarks,
    }
    print(json.dumps(report_dict, indent=2))

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_DIR}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
