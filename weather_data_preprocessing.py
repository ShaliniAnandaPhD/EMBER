import json

def load_weather_data(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def preprocess_weather_data(data):
    main_data = data.get('main', {})
    weather_description = data.get('weather', [{}])[0].get('description', 'No description')
    preprocessed_data = {
        'temperature': main_data.get('temp', None),
        'humidity': main_data.get('humidity', None),
        'pressure': main_data.get('pressure', None),
        'weather_description': weather_description
    }
    return preprocessed_data

def save_preprocessed_data(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
if __name__ == "__main__":
    try:
        raw_data = load_weather_data("weather_data.json")
        preprocessed_data = preprocess_weather_data(raw_data)
        save_preprocessed_data(preprocessed_data, "preprocessed_weather_data.json")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
