import requests
import pandas as pd
import os

def collect_satellite_data():
    # NASA API endpoint for satellite imagery data
    url = "https://api.nasa.gov/planetary/earth/imagery"
    
    # Set API parameters
    params = {
        "lon": "-122.4194",
        "lat": "37.7749",
        "date": "2023-06-01",
        "dim": "0.1",
        "api_key": "YOUR_NASA_API_KEY"
    }
    
    # Send GET request to the API
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the satellite imagery data to a file
        with open("satellite_data.png", "wb") as file:
            file.write(response.content)
        print("Satellite data collected successfully.")
    else:
        print("Failed to collect satellite data.")

def collect_weather_data():
    # OpenWeatherMap API endpoint for weather data
    url = "https://api.openweathermap.org/data/2.5/weather"
    
    # Set API parameters
    params = {
        "q": "San Francisco",
        "appid": "YOUR_OPENWEATHERMAP_API_KEY"
    }
    
    # Send GET request to the API
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract relevant weather data
        weather_data = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        
        # Save the weather data to a CSV file
        df = pd.DataFrame([weather_data])
        df.to_csv("weather_data.csv", index=False)
        print("Weather data collected successfully.")
    else:
        print("Failed to collect weather data.")

def collect_historical_data():
    # URL of the historical wildfire data CSV file
    url = "https://www.kaggle.com/rtatman/188-million-us-wildfires/downloads/FPA_FOD_20170508.sqlite.zip"
    
    # Download the ZIP file
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the ZIP file
        with open("historical_data.zip", "wb") as file:
            file.write(response.content)
        print("Historical data downloaded successfully.")
    else:
        print("Failed to download historical data.")

# Main function to collect data from various sources
def main():
    # Create a directory to store the collected data
    os.makedirs("data", exist_ok=True)
    
    # Change the current working directory to the "data" directory
    os.chdir("data")
    
    # Collect satellite data
    collect_satellite_data()
    
    # Collect weather data
    collect_weather_data()
    
    # Collect historical data
    collect_historical_data()

if __name__ == "__main__":
    main()
