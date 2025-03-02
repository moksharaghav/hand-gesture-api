import os
import shutil
import random

# Paths
dataset_path = "dataset/augmented_lgrr"  # Augmented dataset
train_path = "dataset/train"
test_path = "dataset/test"

# Create train & test directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

SPLIT_RATIO = 0.8  # 80% training, 20% testing

# Process each class folder
for gesture in sorted(os.listdir(dataset_path)):
    gesture_path = os.path.join(dataset_path, gesture)

    if not os.path.isdir(gesture_path) or not gesture.isdigit():  # ✅ Ignore non-numeric files
        continue

    images = os.listdir(gesture_path)
    random.shuffle(images)  # Shuffle images for randomness

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create class folders in train/test directories
    train_class_path = os.path.join(train_path, gesture)
    test_class_path = os.path.join(test_path, gesture)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)

    # Move images to train folder
    for img in train_images:
        src = os.path.join(gesture_path, img)
        dst = os.path.join(train_class_path, img)
        shutil.copy(src, dst)

    # Move images to test folder
    for img in test_images:
        src = os.path.join(gesture_path, img)
        dst = os.path.join(test_class_path, img)
        shutil.copy(src, dst)

    print(f"✅ Split completed for class {gesture}: {len(train_images)} train, {len(test_images)} test")

print("🎉 Data splitting complete! Training and testing sets are ready.")