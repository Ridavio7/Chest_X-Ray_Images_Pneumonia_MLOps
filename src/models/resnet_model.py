import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights


class ResNet18Binary(nn.Module):

    def __init__(self, dropout=0.3):

        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.model(x)