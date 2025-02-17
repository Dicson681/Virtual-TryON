import cv2
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import pyopenpose as op
from op import WrapperPython as opWrapper

def read_last_pair(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) < 1:
            raise ValueError("The file must contain at least one line.")
        return [line.strip().split() for line in lines[-1:]]

def process_image_pair(image_path, cloth_path, output_folder, image_file, cloth_file):
    # Read images
    image = cv2.imread(image_path)
    cloth = cv2.imread(cloth_path)

    # Create subfolders for outputs
    openpose_json_folder = os.path.join(output_folder, 'openpose-json')
    openpose_img_folder = os.path.join(output_folder, 'openpose-img')
    image_parse_folder = os.path.join(output_folder, 'image-parse')
    image_folder = os.path.join(output_folder, 'image')
    cloth_mask_folder = os.path.join(output_folder, 'cloth-mask')
    cloth_folder = os.path.join(output_folder, 'cloth')
    openpose_parse_folder = os.path.join(output_folder, 'openpose_parse')

    # Ensure the directories exist
    os.makedirs(openpose_json_folder, exist_ok=True)
    os.makedirs(openpose_img_folder, exist_ok=True)
    os.makedirs(image_parse_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(cloth_mask_folder, exist_ok=True)
    os.makedirs(cloth_folder, exist_ok=True)
    os.makedirs(openpose_parse_folder, exist_ok=True)

    # Process with OpenPose
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])

    # Save OpenPose JSON in the openpose_json folder
    json_output = datum.to_json()
    json_output_path = os.path.join(openpose_json_folder, f"{os.path.splitext(image_file)[0]}_keypoints.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_output, json_file)

    # Save OpenPose rendered image in the openpose_img folder
    rendered_image_path = os.path.join(openpose_img_folder, f"{os.path.splitext(image_file)[0]}_rendered.png")
    cv2.imwrite(rendered_image_path, datum.cvOutputData)

    # Process with LIP_JPPNet
    model = torch.load("/content/drive/MyDrive/VITON-HD/LIP_JPPNet")
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    image_parse = output[0].cpu().numpy().transpose(1, 2, 0)
    image_parse = (image_parse * 255).astype(np.uint8)
    image_parse_path = os.path.join(openpose_parse_folder, f"{os.path.splitext(image_file)[0]}_image_parse.png")
    cv2.imwrite(image_parse_path, image_parse)

    # Save cloth mask in the cloth_mask folder
    cloth_mask = np.zeros_like(image)
    cloth_mask[cloth == 255] = 255
    cloth_mask_path = os.path.join(cloth_mask_folder, f"{os.path.splitext(cloth_file)[0]}_cloth_mask.png")
    cv2.imwrite(cloth_mask_path, cloth_mask)


dataset_folder = '/content/drive/MyDrive/VITON-HD/datasets/test'
image_folder = os.path.join(dataset_folder, 'images')
cloth_folder = os.path.join(dataset_folder, 'cloth')
output_folder = dataset_folder  # The output will now be saved in datasets/test

test_pairs_file = '/content/drive/MyDrive/VITON-HD/datasets/test_pairs.txt'
image_cloth_pairs = read_last_pair(test_pairs_file)

for image_file, cloth_file in image_cloth_pairs:
    image_path = os.path.join(image_folder, image_file)
    cloth_path = os.path.join(cloth_folder, cloth_file)
    process_image_pair(image_path, cloth_path, output_folder, image_file, cloth_file)