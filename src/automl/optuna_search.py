import optuna
import torch
import torch.nn as nn

from src.models.resnet_model import ResNet18Binary
from src.training.train_utils import train_one_epoch, validate
from src.training.metrics import calculate_metrics
from src.data.data_loader import get_dataloaders


def objective(trial):

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-3,
        log=True
    )

    dropout = trial.suggest_float(
        "dropout",
        0.2,
        0.6
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64]
    )

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18Binary(
        dropout=dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    for _ in range(3):

        _, _, _ = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        _, y_true, y_pred = validate(
            model,
            val_loader,
            criterion,
            device
        )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    return metrics["f1"]


if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=20
    )

    print("\nBEST PARAMS")
    print(study.best_params)

    print("\nBEST F1")
    print(study.best_value)