import os
import shutil

dataset_path = "dataset/processed_lgrr"  # Path where all images are currently stored

# Iterate over all image files
for img_name in os.listdir(dataset_path):
    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        # Extract class label from filename
        parts = img_name.split("_")  # Split filename by underscores (_)
        class_label = parts[1]  # Assuming the second part contains the class (e.g., "00", "01", etc.)

        # Create class folder if it doesn't exist
        class_folder = os.path.join(dataset_path, class_label)
        os.makedirs(class_folder, exist_ok=True)

        # Move image to respective class folder
        src_path = os.path.join(dataset_path, img_name)
        dest_path = os.path.join(class_folder, img_name)
        shutil.move(src_path, dest_path)

print("✅ Images successfully organized into class folders!")