import os
import cv2
import numpy as np

dataset_path = "dataset/lgrr"
processed_path = "dataset/processed_lgrr"

# Create a folder for processed images
os.makedirs(processed_path, exist_ok=True)

IMG_SIZE = 128  # Standard size for resizing

# Iterate through images and process them
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    
    # Read the image
    img = cv2.imread(img_path)
    
    if img is not None:
        # Resize to 128x128
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values (0-1 range)
        img_resized = img_resized / 255.0

        # Save processed image
        save_path = os.path.join(processed_path, img_name)
        cv2.imwrite(save_path, (img_resized * 255).astype(np.uint8))

print("Preprocessing complete! Resized and normalized images saved in 'dataset/processed_lgrr'.")