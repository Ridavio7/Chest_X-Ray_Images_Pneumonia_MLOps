import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class EfficientNetBinary(nn.Module):

    def __init__(self, dropout=0.3):

        super().__init__()

        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        in_features = (
            self.model.classifier[1].in_features
        )

        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.model(x)