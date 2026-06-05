import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score
)

from torch.utils.data import DataLoader

from src.models.efficientnet_model import EfficientNetBinary
from src.data.dataset import XRayDataset
from src.data.transforms import val_transforms
from src.training.metrics import calculate_metrics


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

test_dataset = XRayDataset(
    "data/chest_xray/test",
    transform=val_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
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

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_preds.extend(
            probs.cpu().numpy().reshape(-1)
        )

        all_labels.extend(
            labels.numpy().reshape(-1)
        )

metrics = calculate_metrics(
    all_labels,
    all_preds
)

print("\nTEST METRICS")
print(metrics)

auc = roc_auc_score(
    all_labels,
    all_preds
)

print("ROC-AUC:", auc)

pred_classes = (
    np.array(all_preds) > 0.5
).astype(int)

print(
    classification_report(
        all_labels,
        pred_classes
    )
)

cm = confusion_matrix(
    all_labels,
    pred_classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Test Confusion Matrix"
)

plt.savefig(
    "outputs/test_confusion_matrix.png"
)

plt.close()