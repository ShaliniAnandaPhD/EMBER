import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_weather_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled

def preprocess_historical_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Remove outliers (example: remove data points beyond 3 standard deviations)
    data = data[(data - data.mean()).abs() <= (3 * data.std())]
    
    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled

# Main function to preprocess the collected data
def main():
    # Load the weather data
    weather_data = pd.read_csv("data/weather_data.csv")
    
    # Preprocess the weather data
    preprocessed_weather_data = preprocess_weather_data(weather_data)
    
    # Save the preprocessed weather data
    pd.DataFrame(preprocessed_weather_data).to_csv("data/preprocessed_weather_data.csv", index=False)
    
    # Load the historical data
    historical_data = pd.read_csv("data/historical_data.csv")
    
    # Preprocess the historical data
    preprocessed_historical_data = preprocess_historical_data(historical_data)
    
    # Save the preprocessed historical data
    pd.DataFrame(preprocessed_historical_data).to_csv("data/preprocessed_historical_data.csv", index=False)

if __name__ == "__main__":
    main()
