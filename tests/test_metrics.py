from src.training.metrics import (
    calculate_metrics
)


def test_metrics():

    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.8, 0.2]

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    assert "accuracy" in metrics
    assert "f1" in metrics