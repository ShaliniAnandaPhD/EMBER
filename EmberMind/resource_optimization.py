import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

def optimize_resources(wildfire_probability, resource_data):
    # Prepare the data
    X = resource_data.drop(columns=["resources_deployed"])
    y = resource_data["resources_deployed"]
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the optimal resources based on the wildfire probability
    optimal_resources = model.predict([[wildfire_probability]])
    
    return optimal_resources

# Main function to optimize resource allocation
def main():
    # Load the wildfire probability
    wildfire_probability = np.load("data/wildfire_probability.npy")
    
    # Load the resource data
    resource_data = pd.read_csv("data/resource_data.csv")
    
    # Optimize the resource allocation
    optimal_resources = optimize_resources(wildfire_probability, resource_data)
    
    # Print the optimal resources
    print("Optimal Resources: {}".format(optimal_resources))

if __name__ == "__main__":
    main()
