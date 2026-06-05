from torch.utils.data import DataLoader

from src.data.dataset import XRayDataset
from src.data.transforms import train_transforms, val_transforms


def get_dataloaders(batch_size=32):

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
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader