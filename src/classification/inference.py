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
    """Deserialize the request body into a tensor. Called on each request."""
    if content_type == "application/x-npy":
        # Already resized and normalized by the caller, shape (1, 3, 224, 224)
        array = np.load(io.BytesIO(request_body))
        return torch.tensor(array, dtype=torch.float32)
    elif content_type in ("application/x-image", "image/jpeg", "image/png"):
        # Matches the preprocessing used in training (5_Classification_Model)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        image = Image.open(io.BytesIO(request_body)).convert("RGB").resize(IMAGE_SIZE)
        return transform(image).unsqueeze(0)  # add batch dim
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run inference. Called after input_fn."""
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    with torch.no_grad():
        logits = model(input_data)
    return logits.cpu()


def output_fn(prediction, accept):
    """Serialize the prediction output. Called after predict_fn."""
    if accept == "application/json":
        probabilities = torch.softmax(prediction, dim=1)[0].tolist()
        predicted_class = int(np.argmax(probabilities))
        result = {
            "logits": prediction.tolist(),
            "probabilities": dict(zip(CLASS_NAMES, probabilities)),
            "predicted_class": predicted_class,
            "predicted_label": CLASS_NAMES[predicted_class],
        }
        return json.dumps(result), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
