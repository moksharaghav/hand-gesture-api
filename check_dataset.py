import os
import cv2
import matplotlib.pyplot as plt
import random

dataset_path = "dataset/"
gesture_classes = sorted(os.listdir(dataset_path))

def show_sample_images():
    if len(gesture_classes) == 0:
        print("❌ Error: No gesture classes found in dataset!")
        return
    
    # Ensure we don't sample more gestures than available
    num_samples = min(5, len(gesture_classes))
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i, gesture in enumerate(random.sample(gesture_classes, num_samples)):
        img_folder = os.path.join(dataset_path, gesture)
        img_list = os.listdir(img_folder)

        if not img_list:
            print(f"⚠️ Warning: No images found in {gesture}")
            continue

        img_name = random.choic

