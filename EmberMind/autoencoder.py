import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os

def build_autoencoder(input_dim, encoding_dim):
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    
    # Encoder model
    encoder = Model(input_layer, encoded)
    
    return autoencoder, encoder

def train_autoencoder(data, encoding_dim, epochs, batch_size):
    input_dim = data.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    return autoencoder, encoder

# Main function to train the sparse autoencoder
def main():
    # Load the preprocessed weather data
    weather_data = np.loadtxt("data/preprocessed_weather_data.csv", delimiter=",")
    
    # Load the preprocessed historical data
    historical_data = np.loadtxt("data/preprocessed_historical_data.csv", delimiter=",")
    
    # Concatenate the weather and historical data
    data = np.concatenate((weather_data, historical_data), axis=1)
    
    # Set the parameters
    encoding_dim = 32
    epochs = 50
    batch_size = 32
    
    # Train the autoencoder
    autoencoder, encoder = train_autoencoder(data, encoding_dim, epochs, batch_size)
    
    # Create a directory to store the trained models
    os.makedirs("models", exist_ok=True)
    
    # Save the trained models
    autoencoder.save("models/autoencoder_model.h5")
    encoder.save("models/encoder_model.h5")

if __name__ == "__main__":
    main()
