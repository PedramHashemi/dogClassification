import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def ResnetTuned():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    return model


def model_fn(model_dir):
    logger.info("training the model.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResnetTuned().to(device)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model


def input_fn(request_body, content_type):
    try:
        logger.info("reading input.")
        image = Image.open(io.BytesIO(request_body))
        logger.info("transforming.")
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return t(image).unsqueeze(0)

    except Exception as e:
        logger.error(f"Failed with the following error: {e}.")


def predict_fn(input_data, model):
    logger.info("predicting")
    with torch.no_grad():
        output = model(input_data)
        return torch.argmax(output, dim=1).item()
