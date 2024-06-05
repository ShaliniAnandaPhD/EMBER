import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def load_predictive_model():
    model_path = "models/predictive_model.h5"
    model = load_model(model_path)
    return model

def assess_wildfire_risk(model, encoded_features):
    # Predict the wildfire risk
    wildfire_risk = model.predict(encoded_features)
    
    # Convert the risk to a probability
    wildfire_probability = np.round(wildfire_risk.flatten() * 100, 2)
    
    return wildfire_probability

# Main function to assess wildfire risk
def main():
    # Load the encoded features
    encoded_features = np.load("data/encoded_features.npy")
    
    # Load the trained predictive model
    model = load_predictive_model()
    
    # Assess the wildfire risk
    wildfire_probability = assess_wildfire_risk(model, encoded_features)
    
    # Print the wildfire probability
    print("Wildfire Probability: {}%".format(wildfire_probability))

if __name__ == "__main__":
    main()
