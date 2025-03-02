import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "dataset/processed_lgrr"  # Original dataset
augmented_path = "dataset/augmented_lgrr"  # Augmented dataset

# Create folder for augmented images
os.makedirs(augmented_path, exist_ok=True)

# Define augmentation transformations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

IMG_SIZE = 128  # Target image size

# Process each class folder
for gesture in sorted(os.listdir(dataset_path)):
    gesture_path = os.path.join(dataset_path, gesture)

    if not os.path.isdir(gesture_path):  # Ignore files
        continue
    save_folder = os.path.join(augmented_path, gesture)
    os.makedirs(save_folder, exist_ok=True)

    images = os.listdir(gesture_path)
    print(f"Processing class {gesture}: {len(images)} images")  # ✅ Show progress

    for img_name in images:
        img_path = os.path.join(gesture_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Generate 3 augmented versions per image
            i = 0
            for batch in datagen.flow(img, batch_size=1, save_to_dir=save_folder, save_prefix="aug", save_format="jpg"):
                i += 1
                if i >= 1:  # Stop after creating 3 new images
                    break

    print(f"✅ Augmented class {gesture}")  # ✅ Show completion for each class

print("🎉 Data augmentation complete! Augmented images saved in 'dataset/augmented_lgrr'.")