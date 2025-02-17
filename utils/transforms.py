# utils/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
