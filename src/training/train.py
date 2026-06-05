import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.models.efficientnet_model import EfficientNetBinary
from src.data.dataset import XRayDataset
from src.data.transforms import train_transforms, val_transforms
from src.training.metrics import calculate_metrics
from src.training.train_utils import train_one_epoch, validate

mlflow.set_tracking_uri("http://127.0.0.1:5000")

train_dataset = XRayDataset(
    "data/chest_xray/train",
    transform=train_transforms
)

val_dataset = XRayDataset(
    "data/chest_xray/val",
    transform=val_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = EfficientNetBinary(
    dropout=0.23320338848065456
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=7.889481270400718e-05,
    weight_decay=8.740760705983501e-06
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

num_epochs = 3

train_losses = []
val_losses = []

train_acc = []
val_acc = []

train_f1 = []
val_f1 = []

best_loss = float("inf")

with mlflow.start_run():

    mlflow.log_param("model", "EfficientNet-B0")

    mlflow.log_param(
        "dropout",
        0.23320338848065456
    )

    mlflow.log_param(
        "lr",
        7.889481270400718e-05
    )

    mlflow.log_param(
        "weight_decay",
        8.740760705983501e-06
    )

    mlflow.log_param(
        "batch_size",
        64
    )

    mlflow.log_param(
        "epochs",
        num_epochs
    )

    for epoch in range(num_epochs):

        train_loss, y_train, y_pred_train = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, y_val, y_pred_val = validate(
            model,
            val_loader,
            criterion,
            device
        )

        train_metrics = calculate_metrics(
            y_train,
            y_pred_train
        )

        val_metrics = calculate_metrics(
            y_val,
            y_pred_val
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_acc.append(
            train_metrics["accuracy"]
        )

        val_acc.append(
            val_metrics["accuracy"]
        )

        train_f1.append(
            train_metrics["f1"]
        )

        val_f1.append(
            val_metrics["f1"]
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_efficientnet_optuna.pth"
            )

            print("Best model saved!")

        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch
        )

        mlflow.log_metric(
            "val_accuracy",
            val_metrics["accuracy"],
            step=epoch
        )

        mlflow.log_metric(
            "val_precision",
            val_metrics["precision"],
            step=epoch
        )

        mlflow.log_metric(
            "val_recall",
            val_metrics["recall"],
            step=epoch
        )

        mlflow.log_metric(
            "val_f1",
            val_metrics["f1"],
            step=epoch
        )

        print(f"\nEpoch {epoch + 1}")

        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)

        print("Train Metrics:", train_metrics)
        print("Val Metrics:", val_metrics)

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.figure()

    plt.plot(
        train_losses,
        label="train_loss"
    )

    plt.plot(
        val_losses,
        label="val_loss"
    )

    plt.legend()
    plt.title("Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(
        "outputs/loss.png"
    )

    plt.close()

    plt.figure()

    plt.plot(
        train_acc,
        label="train_acc"
    )

    plt.plot(
        val_acc,
        label="val_acc"
    )

    plt.legend()
    plt.title("Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.savefig(
        "outputs/accuracy.png"
    )

    plt.close()

    plt.figure()

    plt.plot(
        train_f1,
        label="train_f1"
    )

    plt.plot(
        val_f1,
        label="val_f1"
    )

    plt.legend()
    plt.title("F1 Score")

    plt.xlabel("Epoch")
    plt.ylabel("F1")

    plt.savefig(
        "outputs/f1.png"
    )

    plt.close()

    cm = confusion_matrix(
        np.array(y_val).reshape(-1),
        (np.array(y_pred_val).reshape(-1) > 0.5).astype(int)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(
        "Confusion Matrix"
    )

    plt.savefig(
        "outputs/confusion_matrix.png"
    )

    plt.close()

    mlflow.log_artifact(
        "outputs/loss.png"
    )

    mlflow.log_artifact(
        "outputs/accuracy.png"
    )

    mlflow.log_artifact(
        "outputs/f1.png"
    )

    mlflow.log_artifact(
        "outputs/confusion_matrix.png"
    )

    mlflow.log_artifact(
        "best_efficientnet_optuna.pth"
    )