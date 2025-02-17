import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VITONDataset(Dataset):
    def __init__(self, file_path, root_dir="datasets/train/", transform=None):
        """
        Args:
            file_path (str): Path to train_pairs.txt
            root_dir (str): Base dataset directory
            transform (Albumentations.Compose): Data augmentation transforms
        """
        self.pairs = [line.strip().split() for line in open(file_path)]
        self.root_dir = root_dir  # Base directory for dataset

        # Default transformations if none provided
        self.transform = transform or A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Extract filenames from train_pairs.txt
        img_name, mask_name = self.pairs[idx]

        # Construct absolute file paths
        img_path = os.path.join(self.root_dir, "image/", img_name)
        mask_path = os.path.join(self.root_dir, "cloth/", mask_name)  # Adjust if using a different folder

        # Read images
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Error handling for missing files
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Convert mask to correct format if necessary
        if len(mask.shape) == 2:  # If grayscale, add a channel dimension
            mask = np.expand_dims(mask, axis=-1)

        # Apply transformations
        augmented = self.transform(image=image, mask=mask)

        return augmented["image"], augmented["mask"]
