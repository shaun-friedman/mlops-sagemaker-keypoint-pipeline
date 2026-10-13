import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import boto3
import pandas as pd
from PIL import Image

TARGET_W, TARGET_H = 224, 224

KP_COLS = [
    "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
    "mouth_x", "mouth_y", "left_ear_1_x", "left_ear_1_y",
    "left_ear_2_x", "left_ear_2_y", "left_ear_3_x", "left_ear_3_y",
    "right_ear_1_x", "right_ear_1_y", "right_ear_2_x", "right_ear_2_y",
    "right_ear_3_x", "right_ear_3_y",
]

s3 = boto3.client("s3")

def load_image(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return Image.open(BytesIO(obj["Body"].read())).convert("RGB")

def resize_and_save_image(s3_uri, output_dir, target_width, target_height) -> str:
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    img = load_image(bucket, key)
    img_resized = img.resize((target_width, target_height), Image.BILINEAR)
    img_path = os.path.join(output_dir, "_".join(key.split("/")[-2:]))
    img_resized.save(img_path)
    return os.path.basename(img_path)

def transform_keypoint(kp: float, s0: float) -> float:
    return (kp / s0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"])
    parser.add_argument("--sample-size", type=int, required=False, default=None)
    parser.add_argument("--workers", type=int, required=False, default=32)
    args = parser.parse_args()

    csv_files = [
        os.path.join(args.metadata, f)
        for f in os.listdir(args.metadata)
        if f.endswith(".csv") and not f.endswith(".csv.metadata")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.metadata}")

    df = pd.read_csv(csv_files[0])
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.sample_size is not None:
        df = df.sample(args.sample_size, axis=0)
    
    print(f"Processing {len(df)} rows for split=\\'{args.split}\\'")

    os.makedirs(args.output, exist_ok=True)

    invalid = (df["width"] == 0) | (df["height"] == 0)
    for idx, row in df[invalid].iterrows():
        print(f"Skipping row {idx} — invalid dimensions w={row['width']}, h={row['height']}")
    df = df[~invalid].reset_index(drop=True)

    # Downloads are I/O bound, so threads give a large speedup
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        img_filenames = list(pool.map(
            lambda s3_uri: resize_and_save_image(s3_uri, args.output, TARGET_W, TARGET_H),
            df["remote_path"],
        ))

    records = []
    for (_, row), img_filename in zip(df.iterrows(), img_filenames):
        w0, h0 = row["width"], row["height"]
        kp_norm = []
        for col in KP_COLS:
            if col.endswith("_x"):
                kp_norm.append(transform_keypoint(row[col], w0))
            elif col.endswith("_y"):
                kp_norm.append(transform_keypoint(row[col], h0))
            else:
                raise ValueError(f"Unexpected keypoint column: {col}")
        kp_cols_norm = [f"{col}_norm" for col in KP_COLS]
        # Original size and source image are kept so evaluation and monitoring can
        # measure error in original pixels and resend the source image to an endpoint
        records.append({
            "image": img_filename,
            "remote_path": row["remote_path"],
            "width": w0,
            "height": h0,
            **dict(zip(kp_cols_norm, kp_norm)),
        })

    pd.DataFrame(records).to_parquet(
        os.path.join(args.output, "annotations.parquet"), index=False
    )
    print(f"Saved {len(records)} records to annotations.parquet")