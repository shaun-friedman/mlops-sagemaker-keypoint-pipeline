import io
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["non_cat", "cat"]

# Request-image statistics returned with each prediction (see image_stats)
STAT_NAMES = ["width", "height", "aspect_ratio", "brightness", "contrast"]


def image_stats(image):
    """Summary statistics of the request image. Returned with each prediction so the
    data-quality monitor can track input drift without decoding images itself."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def model_fn(model_dir):
    """Load the fine-tuned ResNet-18 classifier. Called once on container startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(
        torch.load(f"{model_dir}/model.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model


def input_fn(request_body, content_type):
    """Deserialize the request body into (tensor, image stats). Called on each request."""
    if content_type == "application/x-npy":
        # Already resized and normalized by the caller, shape (1, 3, 224, 224)
        array = np.load(io.BytesIO(request_body))
        return torch.tensor(array, dtype=torch.float32), None
    elif content_type in ("application/x-image", "image/jpeg", "image/png"):
        # Matches the preprocessing used in training (src/classification/preprocess.py)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return transform(image.resize(IMAGE_SIZE)).unsqueeze(0), image_stats(image)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run inference. Called after input_fn."""
    tensor, stats = input_data
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tensor.to(device))
    return logits.cpu(), stats


def output_fn(prediction, accept):
    """Serialize the prediction output. Called after predict_fn."""
    logits, stats = prediction
    probabilities = torch.softmax(logits, dim=1)[0].tolist()
    predicted_class = int(np.argmax(probabilities))

    if accept == "application/json":
        result = {
            "logits": logits.tolist(),
            "probabilities": dict(zip(CLASS_NAMES, probabilities)),
            "predicted_class": predicted_class,
            "predicted_label": CLASS_NAMES[predicted_class],
        }
        if stats is not None:
            result["input_stats"] = stats
        return json.dumps(result), "application/json"
    elif accept == "text/csv":
        # Columns: predicted_class, cat probability, then STAT_NAMES for image requests.
        # Model Monitor's model-quality job reads the first two columns by index.
        values = [predicted_class, probabilities[1]]
        if stats is not None:
            values += [stats[name] for name in STAT_NAMES]
        return ",".join(str(value) for value in values), "text/csv"
    raise ValueError(f"Unsupported accept type: {accept}")
