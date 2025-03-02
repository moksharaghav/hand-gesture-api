import os
import cv2

dataset_path = "dataset/lgrr"
image_sizes = {}

# Iterate through images in the dataset
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    
    # Read the image
    img = cv2.imread(img_path)
    
    if img is not None:
        h, w, c = img.shape
        size = (h, w)
        if size not in image_sizes:
            image_sizes[size] = 0
        image_sizes[size] += 1

# Print the different image sizes
print("Unique Image Sizes in Dataset:")
for size, count in image_sizes.items():
    print(f"Size {size}: {count} images")