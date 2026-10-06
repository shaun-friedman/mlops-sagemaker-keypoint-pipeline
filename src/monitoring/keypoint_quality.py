"""Model-quality check for the keypoint endpoint.

Model Monitor's model-quality jobs compare one predicted value with one label, so they
cannot score 9 keypoints at once. This job does the equivalent for the keypoint endpoint:
it joins captured predictions with ground truth by inference ID, computes the normalized
mean error (NME), checks it against the baseline constraints, and publishes the results
to CloudWatch. Output files use Model Monitor's statistics/constraint_violations layout.
"""
import argparse
import base64
import glob
import json
import os
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np

KP_NAMES = [
    "left_eye", "right_eye", "mouth",
    "left_ear_1", "left_ear_2", "left_ear_3",
    "right_ear_1", "right_ear_2", "right_ear_3",
]


def read_jsonl(directory):
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.jsonl"), recursive=True)):
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def decode(captured):
    """Payload of a captured endpoint output or a ground-truth record, as a dict."""
    data = captured["data"]
    if captured.get("encoding") == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    return json.loads(data) if isinstance(data, str) else data


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def keypoint_nme(pred_norm, gt_norm, width, height):
    """Per-keypoint pixel error divided by the ground-truth inter-ocular distance.
    Same definition as src/keypoint_regression/evaluation.py."""
    scale = np.array([width, height], dtype=float)
    pred_px = np.asarray(pred_norm, dtype=float).reshape(-1, 2) * scale
    gt_px = np.asarray(gt_norm, dtype=float).reshape(-1, 2) * scale
    inter_ocular = np.linalg.norm(gt_px[0] - gt_px[1])
    if inter_ocular == 0:
        return np.full(len(KP_NAMES), np.nan)
    return np.linalg.norm(pred_px - gt_px, axis=1) / inter_ocular


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=str, default="/opt/ml/processing/input/capture")
    parser.add_argument("--ground-truth-dir", type=str, default="/opt/ml/processing/input/ground_truth")
    parser.add_argument("--constraints", type=str, default="/opt/ml/processing/input/constraints/constraints.json")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/report")
    # "auto" means the last --lookback-hours hours, which is what scheduled runs use
    parser.add_argument("--start-time", type=str, default="auto")
    parser.add_argument("--end-time", type=str, default="auto")
    parser.add_argument("--lookback-hours", type=int, default=1)
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--metric-namespace", type=str, default="")
    args = parser.parse_args()

    end = datetime.now(timezone.utc) if args.end_time == "auto" else parse_time(args.end_time)
    start = end - timedelta(hours=args.lookback_hours) if args.start_time == "auto" else parse_time(args.start_time)
    print(f"Analysis window: {start.isoformat()} to {end.isoformat()}")

# ---------------------------------------------------------------------------
# Join captured predictions with ground truth
# ---------------------------------------------------------------------------
    predictions = {}
    for record in read_jsonl(args.capture_dir):
        metadata = record["eventMetadata"]
        inference_id = metadata.get("inferenceId")
        if inference_id is None or not start <= parse_time(metadata["inferenceTime"]) < end:
            continue
        output = decode(record["captureData"]["endpointOutput"])
        predictions[inference_id] = [output[kp][axis] for kp in KP_NAMES for axis in ("x", "y")]

    ground_truth = {
        record["eventMetadata"]["eventId"]: decode(record["groundTruthData"])
        for record in read_jsonl(args.ground_truth_dir)
    }

    matched = sorted(set(predictions) & set(ground_truth))
    print(f"Captured predictions in window: {len(predictions)} | with ground truth: {len(matched)}")

# ---------------------------------------------------------------------------
# NME statistics
# ---------------------------------------------------------------------------
    nme = np.array([
        keypoint_nme(predictions[i], ground_truth[i]["keypoints_norm"],
                     ground_truth[i]["width"], ground_truth[i]["height"])
        for i in matched
    ]).reshape(-1, len(KP_NAMES))
    nme = nme[~np.isnan(nme).any(axis=1)]
    nme_per_image = nme.mean(axis=1)

    metrics = {}
    if len(nme_per_image):
        metrics = {
            "nme_mean": float(nme_per_image.mean()),
            "nme_median": float(np.median(nme_per_image)),
            "nme_p95": float(np.percentile(nme_per_image, 95)),
            "failure_rate_at_0_1": float((nme_per_image > 0.1).mean()),
        }

    statistics = {
        "version": 0.0,
        "dataset": {
            "item_count": int(len(nme_per_image)),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
        },
        "keypoint_metrics": {name: {"value": value} for name, value in metrics.items()},
        "per_keypoint_nme": {
            name: {"value": float(value)} for name, value in zip(KP_NAMES, nme.mean(axis=0))
        } if len(nme) else {},
    }

# ---------------------------------------------------------------------------
# Constraint check
# ---------------------------------------------------------------------------
    with open(args.constraints) as f:
        constraints = json.load(f)["keypoint_constraints"]

    violations = []
    for name, constraint in constraints.items():
        if name in metrics and constraint["comparison_operator"] == "GreaterThanThreshold" \
                and metrics[name] > constraint["threshold"]:
            violations.append({
                "metric_name": name,
                "constraint_check_type": "GreaterThanThreshold",
                "description": f"Metric {name} with {metrics[name]:.4f} was GreaterThanThreshold "
                               f"'{constraint['threshold']:.4f}'",
            })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
        json.dump(statistics, f, indent=2)
    with open(os.path.join(args.output_dir, "constraint_violations.json"), "w") as f:
        json.dump({"violations": violations}, f, indent=2)

    print(json.dumps(statistics, indent=2))
    print(json.dumps({"violations": violations}, indent=2))

# ---------------------------------------------------------------------------
# CloudWatch metrics
# ---------------------------------------------------------------------------
    if args.metric_namespace:
        dimensions = [{"Name": "Endpoint", "Value": args.endpoint_name}]
        metric_data = [
            {"MetricName": name, "Dimensions": dimensions, "Value": value}
            for name, value in metrics.items()
        ]
        metric_data += [
            {"MetricName": "matched_records", "Dimensions": dimensions, "Value": float(len(nme_per_image))},
            {"MetricName": "constraint_violations", "Dimensions": dimensions, "Value": float(len(violations))},
        ]
        boto3.client("cloudwatch", region_name=args.region).put_metric_data(
            Namespace=args.metric_namespace, MetricData=metric_data
        )
        print(f"Published {len(metric_data)} metrics to {args.metric_namespace}")
