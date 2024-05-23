# self_supervised_satellite_segmentation.py

import torch
import numpy as np
from torchvision import transforms
from transformers import ViTForImageSegmentation
from PIL import Image
import os

def load_satellite_images(data_dir):
    """
    Load satellite images from a directory.

    Args:
        data_dir (str): Directory containing the satellite images.

    Returns:
        list: List of loaded satellite images as PIL Image objects.
    """
    try:
        image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg") or f.endswith(".png")]
        images = []
        for file in image_files:
            image_path = os.path.join(data_dir, file)
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        return images

    except FileNotFoundError:
        print(f"Error: Directory not found: {data_dir}")
        return None

    except Exception as e:
        print(f"Error: An unexpected error occurred during image loading: {str(e)}")
        return None

def preprocess_images(images, image_size=224):
    """
    Preprocess the satellite images for input to the segmentation model.

    Args:
        images (list): List of satellite images as PIL Image objects.
        image_size (int): Desired size of the preprocessed images (default: 224).

    Returns:
        torch.Tensor: Preprocessed images as a tensor.
    """
    try:
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        preprocessed_images = []
        for image in images:
            preprocessed_image = preprocess(image)
            preprocessed_images.append(preprocessed_image)

        preprocessed_images = torch.stack(preprocessed_images)
        return preprocessed_images

    except Exception as e:
        print(f"Error: An unexpected error occurred during image preprocessing: {str(e)}")
        return None

def segment_satellite_images(images, model_name="facebook/vit-mae-base"):
    """
    Segment satellite images using a self-supervised learning model.

    Args:
        images (torch.Tensor): Preprocessed satellite images as a tensor.
        model_name (str): Name of the pre-trained self-supervised learning model (default: "facebook/vit-mae-base").

    Returns:
        torch.Tensor: Segmented images as a tensor.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ViTForImageSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()

        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            segments = outputs.logits.argmax(dim=1)

        return segments.cpu()

    except Exception as e:
        print(f"Error: An unexpected error occurred during image segmentation: {str(e)}")
        return None

def save_segmented_images(segments, output_dir):
    """
    Save the segmented images to a directory.

    Args:
        segments (torch.Tensor): Segmented images as a tensor.
        output_dir (str): Directory to save the segmented images.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        num_images = segments.shape[0]
        for i in range(num_images):
            segment = segments[i].numpy()
            segment_image = Image.fromarray(segment.astype(np.uint8))
            segment_image.save(os.path.join(output_dir, f"segment_{i+1}.png"))

        print(f"Segmented images saved to: {output_dir}")

    except Exception as e:
        print(f"Error: An unexpected error occurred during saving segmented images: {str(e)}")

def visualize_segmentations(images, segments):
    """
    Visualize the original images and their corresponding segmentations.

    Args:
        images (list): List of original satellite images as PIL Image objects.
        segments (torch.Tensor): Segmented images as a tensor.
    """
    try:
        import matplotlib.pyplot as plt

        num_images = len(images)
        rows = num_images // 2
        cols = 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5*rows))
        axes = axes.flatten()

        for i in range(num_images):
            axes[i*2].imshow(images[i])
            axes[i*2].set_title(f"Original Image {i+1}")
            axes[i*2].axis("off")

            segment = segments[i].numpy()
            axes[i*2+1].imshow(segment, cmap="viridis")
            axes[i*2+1].set_title(f"Segmented Image {i+1}")
            axes[i*2+1].axis("off")

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Error: matplotlib library is required for visualization.")

    except Exception as e:
        print(f"Error: An unexpected error occurred during visualization: {str(e)}")

def main():
    data_dir = "path/to/satellite/images"
    output_dir = "path/to/segmented/images"

    images = load_satellite_images(data_dir)

    if images is None:
        print("Error: Failed to load satellite images.")
        return

    preprocessed_images = preprocess_images(images)

    if preprocessed_images is None:
        print("Error: Failed to preprocess satellite images.")
        return

    segments = segment_satellite_images(preprocessed_images)

    if segments is None:
        print("Error: Failed to segment satellite images.")
        return

    save_segmented_images(segments, output_dir)
    visualize_segmentations(images, segments)

if __name__ == "__main__":
    main()
