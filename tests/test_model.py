import torch

from src.models.efficientnet_model import (
    EfficientNetBinary
)


def test_model_output_shape():

    model = EfficientNetBinary()

    x = torch.randn(
        1,
        3,
        224,
        224
    )

    output = model(x)

    assert output.shape == (1, 1)