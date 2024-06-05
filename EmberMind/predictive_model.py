import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import os

def build_predictive_model(input_dim, encoding_dim):
    # Input layer
    input_layer = Input(shape=(None, input_dim))
    
    # LSTM layer
    lstm_layer = LSTM(encoding_dim, return_sequences=True)(input_layer)
    
    # Output layer
    output_layer = Dense(1, activation='sigmoid')(lstm_layer)
    
    # Predictive model
    model = Model(input_layer, output_layer)
    
    return model

def train_predictive_model(data, encoding_dim, epochs, batch_size):
    input_dim = data.shape[2]
    model = build_predictive_model(input_dim, encoding_dim)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, data[:, :, 0], epochs=epochs, batch_size=batch_size, shuffle=True)
    
    return model

# Main function to train the predictive model
def main():
    # Load the encoded features
    encoded_features = np.load("data/encoded_features.npy")
    
    # Set the parameters
    encoding_dim = 32
    epochs = 50
    batch_size = 32
    
    # Train the predictive model
    model = train_predictive_model(encoded_features, encoding_dim, epochs, batch_size)
    
    # Create a directory to store the trained model
    os.makedirs("models", exist_ok=True)
    
    # Save the trained model
    model.save("models/predictive_model.h5")

if __name__ == "__main__":
    main()
