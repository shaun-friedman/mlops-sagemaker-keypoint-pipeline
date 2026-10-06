import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from PIL import Image

TARGET_W, TARGET_H = 224, 224

s3 = boto3.client("s3")

def load_image_bytes(s3_uri):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def image_stats(image):
    """Same statistics as image_stats in src/classification/inference.py."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }

def resize_and_save_image(s3_uri, output_path):
    """Save the 224x224 training image and return statistics of the original image."""
    image_bytes = load_image_bytes(s3_uri)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    # Statistics of the original image, used by the image-statistics benchmark in evaluation.py
    stats = {**image_stats(img), "file_size": len(image_bytes)}
    # Same resize call as src/classification/inference.py, so training sees what serving sees.
    # PNG avoids a second round of JPEG compression that serving never applies.
    img.resize((TARGET_W, TARGET_H)).save(output_path, format="PNG")
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"])
    parser.add_argument("--sample-size", type=int, required=False, default=None)
    parser.add_argument("--workers", type=int, required=False, default=32)
    args = parser.parse_args()

    manifest_files = [
        os.path.join(args.manifest, f)
        for f in os.listdir(args.manifest)
        if f.endswith((".pqt", ".parquet"))
    ]
    if not manifest_files:
        raise FileNotFoundError(f"No parquet manifest found in {args.manifest}")

    df = pd.concat([pd.read_parquet(f) for f in manifest_files])
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.sample_size is not None and args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=42).reset_index(drop=True)

    print(f"Processing {len(df)} rows for split='{args.split}'")
    print(df["label"].value_counts().to_string())

    os.makedirs(args.output, exist_ok=True)
    filenames = [f"{idx:06d}.png" for idx in range(len(df))]
    output_paths = [os.path.join(args.output, f) for f in filenames]

    # Downloads are I/O bound, so threads give a large speedup
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        stats = pd.DataFrame(list(pool.map(resize_and_save_image, df["remote_path"], output_paths)))

    pd.DataFrame({
        "image": filenames,
        "label": df["label"].astype(int),
        "source": df["source"] if "source" in df else "unknown",
        "remote_path": df["remote_path"],
        **{column: stats[column] for column in stats.columns},
    }).to_parquet(os.path.join(args.output, "labels.parquet"), index=False)
    print(f"Saved {len(df)} records to labels.parquet")
