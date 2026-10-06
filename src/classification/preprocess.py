import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import boto3
import pandas as pd
from PIL import Image

TARGET_W, TARGET_H = 224, 224

s3 = boto3.client("s3")

def load_image(s3_uri):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return Image.open(BytesIO(obj["Body"].read())).convert("RGB")

def resize_and_save_image(s3_uri, output_path):
    # Same resize call as src/classification/inference.py, so training sees what serving sees.
    # PNG avoids a second round of JPEG compression that serving never applies.
    img = load_image(s3_uri).resize((TARGET_W, TARGET_H))
    img.save(output_path, format="PNG")

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
        list(pool.map(resize_and_save_image, df["remote_path"], output_paths))

    pd.DataFrame({
        "image": filenames,
        "label": df["label"].astype(int),
        "remote_path": df["remote_path"],
    }).to_parquet(os.path.join(args.output, "labels.parquet"), index=False)
    print(f"Saved {len(df)} records to labels.parquet")
