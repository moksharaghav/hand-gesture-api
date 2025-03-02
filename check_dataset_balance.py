import os

dataset_path = "dataset/processed_lgrr"  # Path where preprocessed images are stored
gesture_classes = [g for g in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, g))]

# Count images in each class
for gesture in gesture_classes:
    gesture_path = os.path.join(dataset_path, gesture)
    num_images = len(os.listdir(gesture_path))
    print(f"Class {gesture}: {num_images} images")