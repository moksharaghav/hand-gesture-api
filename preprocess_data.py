import os
import cv2
import numpy as np

# Define dataset path
dataset_path = "dataset/"
IMG_SIZE = 128  # Image size (128x128 pixels)

X = []  # List to store image data
y = []  # List to store labels (gesture categories)

# Get all gesture categories
gesture_classes = sorted(os.listdir(dataset_path))
print("Processing Classes:", gesture_classes)

for label, gesture in enumerate(gesture_classes):
    gesture_folder = os.path.join(dataset_path, gesture)
    
    # Loop through each image in the category folder
    for img_name in os.listdir(gesture_folder):
        img_path = os.path.join(gesture_folder, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip if the image is corrupted

        # Resize image to 128x128
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values (scale between 0-1)
        img = img / 255.0

        # Append to dataset
        X.append(img)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save processed data for training
np.save("X_train.npy", X)
np.save("y_train.npy", y)

print(f"✅ Data Preprocessing Complete! Processed {len(X)} images.")
