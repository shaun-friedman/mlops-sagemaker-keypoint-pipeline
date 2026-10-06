"""Model Monitor record preprocessor for the keypoint endpoint's data-quality schedule.

Turns one captured request/response into the features in the input/prediction drift
baseline built in 6_Monitoring: the 18 predicted coordinates, the predicted eye distance,
and the request-image statistics the endpoint returns (src/keypoint_regression/inference.py).
"""
import base64
import json
import math

KP_NAMES = [
    "left_eye", "right_eye", "mouth",
    "left_ear_1", "left_ear_2", "left_ear_3",
    "right_ear_1", "right_ear_2", "right_ear_3",
]
STAT_NAMES = ["width", "height", "aspect_ratio", "brightness", "contrast"]
INTEGER_FEATURES = {"height", "width"}

# Alphabetical, matching the baseline dataset's column order
FEATURES = sorted(
    [f"{kp}_{axis}_pred" for kp in KP_NAMES for axis in ("x", "y")]
    + ["pred_inter_ocular"]
    + STAT_NAMES
)


def _decode(captured):
    data = captured.data
    if captured.encoding == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    return data


def preprocess_handler(inference_record):
    result = json.loads(_decode(inference_record.endpoint_output))

    values = {f"{kp}_{axis}_pred": result[kp][axis] for kp in KP_NAMES for axis in ("x", "y")}
    values["pred_inter_ocular"] = math.hypot(
        result["left_eye"]["x"] - result["right_eye"]["x"],
        result["left_eye"]["y"] - result["right_eye"]["y"],
    )
    values.update(result["input_stats"])

    return {
        name: int(values[name]) if name in INTEGER_FEATURES else float(values[name])
        for name in FEATURES
    }
