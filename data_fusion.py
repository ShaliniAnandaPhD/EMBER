import json
import numpy as np
import cv2

def load_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def load_image(file_name):
    return cv2.imread(file_name)

def fuse_data(satellite_image, weather_data, social_media_data):
    fused_data = {
        'satellite_image': satellite_image.tolist(),
        'temperature': weather_data['temperature'],
        'humidity': weather_data['humidity'],
        'pressure': weather_data['pressure'],
        'weather_description': weather_data['weather_description'],
        'social_media_texts': [tweet['preprocessed_text'] for tweet in social_media_data]
    }
    return fused_data

def save_fused_data(fused_data, file_name):
    with open(file_name, 'w') as file:
        json.dump(fused_data, file, indent=4)

# Example usage
if __name__ == "__main__":
    try:
        satellite_image = load_image("processed_image_0.jpg")
        weather_data = load_json("preprocessed_weather_data.json")
        social_media_data = load_json("preprocessed_tweets.json")

        fused_data = fuse_data(satellite_image, weather_data, social_media_data)
        save_fused_data(fused_data, "fused_data.json")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
