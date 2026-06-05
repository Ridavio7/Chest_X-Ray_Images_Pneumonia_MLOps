import os
import cv2
from torch.utils.data import Dataset


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image):
        return self.clahe.apply(image)


class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_clahe=True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_clahe = use_clahe
        self.clahe = CLAHETransform()

        self.images = []
        self.labels = []

        for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_path = os.path.join(root_dir, class_name)

            for img_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        
        # CLAHE
        if self.use_clahe:
            image = self.clahe(image)

        # grayscale -> RGB для ResNet
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        # Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]

        return image, label
