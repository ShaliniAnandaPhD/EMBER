import cv2
import numpy as np
from skimage import exposure

def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def augment_image(image):
    augmented_images = [image]
    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)
    return augmented_images

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found")
    
    image = resize_image(image)
    image = normalize_image(image)
    images = augment_image(image)
    
    return images

# Example usage
if __name__ == "__main__":
    try:
        processed_images = preprocess_image("path_to_satellite_image.jpg")
        for i, img in enumerate(processed_images):
            cv2.imwrite(f"processed_image_{i}.jpg", img)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
