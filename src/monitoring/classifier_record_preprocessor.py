"""Model Monitor record preprocessor for the classifier endpoint's data-quality schedule.

Turns one captured request/response into the features in the input/prediction drift
baseline built in 6_Monitoring. Requests are images, which Model Monitor cannot analyze,
so the features come from the response: the predicted class and cat probability, plus
the request-image statistics the endpoint returns (src/classification/inference.py).
"""
import base64
import json

# Alphabetical, matching the baseline dataset's column order
FEATURES = ["aspect_ratio", "brightness", "contrast", "height", "prediction", "probability", "width"]
INTEGER_FEATURES = {"height", "prediction", "width"}

# Column order of the endpoint's CSV response
CSV_COLUMNS = ["prediction", "probability", "width", "height", "aspect_ratio", "brightness", "contrast"]


def _decode(captured):
    data = captured.data
    if captured.encoding == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    return data.strip()


def preprocess_handler(inference_record):
    output = _decode(inference_record.endpoint_output)

    if output.startswith("{"):
        result = json.loads(output)
        values = {
            "prediction": result["predicted_class"],
            "probability": result["probabilities"]["cat"],
            **result["input_stats"],
        }
    else:
        values = dict(zip(CSV_COLUMNS, output.split(",")))

    return {
        name: int(float(values[name])) if name in INTEGER_FEATURES else float(values[name])
        for name in FEATURES
    }
