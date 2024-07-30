import cv2
import numpy as np
import os

# Function to resize an image
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Function to add Gaussian noise to an RGB image using cv2
def add_gaussian_noise(image, mean, std_deviation):
    noise = np.random.normal(mean, std_deviation, image.shape).astype(np.uint8)
    noised_image = cv2.add(image, noise)
    return noised_image

# Function to add Impulse noise to an image
def add_impulse_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Add Salt noise
    salt_mask = np.random.random(image.shape[:2])
    noisy_image[salt_mask < salt_prob] = 255

    # Add Pepper noise
    pepper_mask = np.random.random(image.shape[:2])
    noisy_image[pepper_mask < pepper_prob] = 0

    return noisy_image

# Function to add Salt-and-Pepper noise to an image
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Add Salt noise
    salt_mask = np.random.random(image.shape[:2])
    noisy_image[salt_mask < salt_prob] = 255

    # Add Pepper noise
    pepper_mask = np.random.random(image.shape[:2])
    noisy_image[pepper_mask < pepper_prob] = 0

    return noisy_image

# Function to add Quantization noise to an image
def add_quantization_noise(image, levels):
    noisy_image = np.copy(image)
    quant_levels = levels
    noisy_image = (np.floor(noisy_image / 255 * quant_levels) * (255 / quant_levels)).astype(np.uint8)
    return noisy_image

# Read the image filename and number of noise levels
n = int(input("Enter the number of noise levels for each type of noise: "))

# Generate n random values for standard deviation for Gaussian noise
mean = 0
std_deviations = np.random.randint(10, 100, n)  # Random values between 10 and 100

# Generate n random values for salt and pepper probabilities
salt_probs = np.random.uniform(0.01, 0.1, n)
pepper_probs = np.random.uniform(0.01, 0.1, n)

# Generate n random values for quantization levels
quantization_levels = np.random.randint(2, 10, n)  # Random levels between 2 and 10

# Ensure the output directory exists
output_dir = 'noise_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the images_background folder
input_dir = 'images_background'
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)  # Load the image in RGB

        # Resize image to 1920x1080
        img = resize_image(img, 1920, 1080)

        # Save the original image
        original_output_path = os.path.join(output_dir, f'original_{filename}')
        cv2.imwrite(original_output_path, img)
        print(f"Original image saved to {original_output_path}")

        # Apply Gaussian Noise and save the images
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        for std_dev in std_deviations:
            img_noised = add_gaussian_noise(img, mean, std_dev)
            output_path = os.path.join(output_dir, f'img_gaussian_noised_{base_filename}_{std_dev}.jpg')
            cv2.imwrite(output_path, img_noised)
            print(f"Image with Gaussian noise (σ={std_dev}) saved to {output_path}")

        # Apply Impulse Noise and save the images
        for salt_prob, pepper_prob in zip(salt_probs, pepper_probs):
            img_noised = add_impulse_noise(img, salt_prob, pepper_prob)
            output_path = os.path.join(output_dir, f'img_impulse_noised_{base_filename}_{salt_prob}_{pepper_prob}.jpg')
            cv2.imwrite(output_path, img_noised)
            print(f"Image with Impulse noise (salt_prob={salt_prob}, pepper_prob={pepper_prob}) saved to {output_path}")

        # Apply Salt-and-Pepper Noise and save the images
        for salt_prob, pepper_prob in zip(salt_probs, pepper_probs):
            img_noised = add_salt_and_pepper_noise(img, salt_prob, pepper_prob)
            output_path = os.path.join(output_dir, f'img_salt_and_pepper_noised_{base_filename}_{salt_prob}_{pepper_prob}.jpg')
            cv2.imwrite(output_path, img_noised)
            print(f"Image with Salt-and-Pepper noise (salt_prob={salt_prob}, pepper_prob={pepper_prob}) saved to {output_path}")

        # Apply Quantization Noise and save the images
        for quant_level in quantization_levels:
            img_noised = add_quantization_noise(img, quant_level)
            output_path = os.path.join(output_dir, f'img_quantization_noised_{base_filename}_{quant_level}.jpg')
            cv2.imwrite(output_path, img_noised)
            print(f"Image with Quantization noise (levels={quant_level}) saved to {output_path}")
