import cv2
import numpy as np
import os


# Function to resize the small image while maintaining aspect ratio
def resize_image(small_img, new_height):
    # Get original dimensions
    original_height, original_width = small_img.shape[:2]

    # Calculate new dimensions
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(small_img, (new_width, new_height))

    return resized_img


# Function to impose a small image onto a larger background image at random positions
def impose_images(background_img, small_imgs, n, output_folder_imgs, output_folder_annotations, big_img_name):
    img_height, img_width, _ = background_img.shape

    # Iterate over each small image
    for small_idx, small_img in enumerate(small_imgs):
        # Resize the small image to height 50 while maintaining aspect ratio
        small_img = resize_image(small_img, 40)
        small_height, small_width, _ = small_img.shape[:3]  # Ensure we only get height, width

        # Iterate over each instance
        for idx in range(n):
            # Create a copy of the background image
            imposed_img = background_img.copy()

            # Randomly select position for pasting the small image
            x = np.random.randint(0, img_width - small_width)
            y = np.random.randint(0, img_height - small_height)

            # Check if the small image has an alpha channel (transparency)
            if small_img.shape[2] == 4:  # Assuming RGBA format
                # Split the small image into channels
                small_img_rgb = small_img[:, :, :3]
                alpha_mask = small_img[:, :, 3] / 255.0  # Normalize alpha channel

                # Blend small image with background using alpha mask
                for c in range(3):
                    imposed_img[y:y + small_height, x:x + small_width, c] = (
                            alpha_mask * small_img_rgb[:, :, c] +
                            (1 - alpha_mask) * imposed_img[y:y + small_height, x:x + small_width, c]
                    )

            else:
                # Paste the small image onto the background image
                imposed_img[y:y + small_height, x:x + small_width] = small_img

            # Calculate YOLO format coordinates (normalized)
            x_center = (x + small_width / 2) / img_width
            y_center = (y + small_height / 2) / img_height
            width_norm = small_width / img_width
            height_norm = small_height / img_height

            # Create annotation in YOLO v8 format
            annotation = f"{small_idx} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

            # Save imposed image with big image name and index
            imposed_img_name = f'{big_img_name}_imposed_{small_idx}_{idx}.jpg'
            imposed_img_path = os.path.join(output_folder_imgs, imposed_img_name)
            cv2.imwrite(imposed_img_path, imposed_img)
            print(f"Imposed image {small_idx}_{idx} saved to {imposed_img_path}")

            # Save annotation in a text file with the same name
            annotation_name = f'{big_img_name}_imposed_{small_idx}_{idx}.txt'
            annotation_path = os.path.join(output_folder_annotations, annotation_name)
            with open(annotation_path, 'w') as f:
                f.write(annotation + '\n')
            print(f"Annotation {small_idx}_{idx} saved to {annotation_path}")


# Function to iterate over images in a folder and apply impose_images function
def process_images_in_folder(folder_path, small_imgs, n, output_folder_imgs, output_folder_annotations):
    # Iterate over images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")

            # Load background image
            background_img = cv2.imread(image_path)

            # Get the base name of the big image without extension
            big_img_name = os.path.splitext(filename)[0]

            # Perform image imposition for each small image
            impose_images(background_img, small_imgs, n, output_folder_imgs, output_folder_annotations, big_img_name)


# Load the small images
small_img_paths = ['logo.png', 'image_logo.png']
small_imgs = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in small_img_paths]  # Load with alpha channel if present

# Number of times to impose each small image
n = 1

# Output folders for saving imposed images and annotations
output_folder_imgs = 'imposed_images'
if not os.path.exists(output_folder_imgs):
    os.makedirs(output_folder_imgs)

output_folder_annotations = 'annotations'
if not os.path.exists(output_folder_annotations):
    os.makedirs(output_folder_annotations)

# Folder containing images to process
folder_path = 'noise_images'

# Process images in the folder
process_images_in_folder(folder_path, small_imgs, n, output_folder_imgs, output_folder_annotations)
