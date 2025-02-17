# scripts/inference.py
import torch
import cv2
import numpy as np
from models.segmentation import SegmentationModel
from models.geometric_matching import GeometricMatchingModel
from models.appearance_flow import AppearanceFlow
from utils.transforms import get_transforms

# Load models
seg_model = SegmentationModel().cuda()
seg_model.load_state_dict(torch.load("checkpoints/seg_model.pth"))
seg_model.eval()

gmm_model = GeometricMatchingModel().cuda()
gmm_model.load_state_dict(torch.load("checkpoints/gmm_model.pth"))
gmm_model.eval()

alias_model = AppearanceFlow().cuda()
alias_model.load_state_dict(torch.load("checkpoints/alias_model.pth"))
alias_model.eval()

# Load test dataset
def load_image(image_path):
    image = cv2.imread(image_path)
    transform = get_transforms()
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0).cuda()

def run_inference():
    test_images = ["datasets/test/" + line.strip().split()[0] for line in open("datasets/test_pairs.txt")]

    for img_path in test_images:
        image = load_image(img_path)

        # Segmentation
        seg_output = seg_model(image)
        segmented_mask = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()

        # Geometric Matching
        gmm_output = gmm_model(image)

        # Appearance Flow
        final_output = alias_model(image)

        # Save output
        cv2.imwrite(f"datasets/test/output/{img_path.split('/')[-1]}", final_output.detach().cpu().numpy() * 255)

        print(f"Inference complete for {img_path}")

if __name__ == "__main__":
    run_inference()
