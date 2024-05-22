import cv2
import numpy as np

def extract_edges(image):
    return cv2.Canny(image, 100, 200)

def extract_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found")

    edges = extract_edges(image)
    histogram = extract_histogram(image)
    
    return edges, histogram

# Example usage
if __name__ == "__main__":
    try:
        edges, histogram = extract_features("path_to_satellite_image.jpg")
        cv2.imwrite("edges.jpg", edges)
        np.savetxt("histogram.csv", histogram, delimiter=",")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
