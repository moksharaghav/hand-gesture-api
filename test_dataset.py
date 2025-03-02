import os

# Define dataset path
dataset_path = "dataset/lgrr"  # Updated for "lgrr"

# Check if the dataset folder exists
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
else:
    gesture_classes = sorted(os.listdir(dataset_path))
    print("Gesture Classes:", gesture_classes)

    # Check if each folder contains images
    for gesture in gesture_classes:
        gesture_path = os.path.join(dataset_path, gesture)
        if os.path.isdir(gesture_path):
            images = os.listdir(gesture_path)
            print(f"{gesture}: {len(images)} images")
