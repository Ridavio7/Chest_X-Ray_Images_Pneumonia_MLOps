import sys
import torch
import numpy as np

from PIL import Image

from src.models.efficientnet_model import EfficientNetBinary
from src.data.transforms import val_transforms


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = EfficientNetBinary(
    dropout=0.23320338848065456
).to(device)

model.load_state_dict(
    torch.load(
        "best_efficientnet_optuna.pth",
        map_location=device
    )
)

model.eval()


def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = np.array(image)

    transformed = val_transforms(
        image=image
    )

    tensor = transformed["image"] \
        .unsqueeze(0) \
        .to(device)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(
            output
        ).item()

    prediction = (
        "PNEUMONIA"
        if probability > 0.5
        else "NORMAL"
    )

    print()
    print("Prediction:", prediction)
    print(
        "Probability:",
        round(probability * 100, 2),
        "%"
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python -m src.inference.predict image.jpg"
        )

        sys.exit()

    predict(
        sys.argv[1]
    )