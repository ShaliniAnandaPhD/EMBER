import requests
import json

API_KEY = "your_api_key"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def fetch_weather_data(city):
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch weather data: {response.status_code}")
    
    return response.json()

def save_weather_data(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
if __name__ == "__main__":
    try:
        city = "San Francisco"
        weather_data = fetch_weather_data(city)
        save_weather_data(weather_data, "weather_data.json")
    except ConnectionError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
