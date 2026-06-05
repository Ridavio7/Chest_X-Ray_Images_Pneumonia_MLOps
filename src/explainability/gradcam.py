import sys
import cv2
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

activations = []
gradients = []


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


target_layer = model.model.features[-1]

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)

image_path = sys.argv[1]

image_pil = Image.open(
    image_path
).convert("RGB")

image_np = np.array(image_pil)

transformed = val_transforms(
    image=image_np
)

tensor = transformed["image"].unsqueeze(0).to(device)

output = model(tensor)

model.zero_grad()

output.backward()

acts = activations[0].detach().cpu().numpy()[0]
grads = gradients[0].detach().cpu().numpy()[0]

weights = np.mean(
    grads,
    axis=(1, 2)
)

cam = np.zeros(
    acts.shape[1:],
    dtype=np.float32
)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)

cam = cv2.resize(
    cam,
    (
        image_np.shape[1],
        image_np.shape[0]
    )
)

cam = cam - cam.min()

if cam.max() > 0:
    cam = cam / cam.max()

heatmap = np.uint8(
    255 * cam
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    image_np,
    0.6,
    heatmap,
    0.4,
    0
)

cv2.imwrite(
    "outputs/gradcam_result.jpg",
    cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )
)

print(
    "Grad-CAM saved to outputs/gradcam_result.jpg"
)