# efficient_transformer_forecasting.py

import torch
import numpy as np
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_weather_data(file_path):
    """
    Load and preprocess the weather data.

    Args:
        file_path (str): Path to the CSV file containing weather data.

    Returns:
        tuple: A tuple containing the preprocessed data and labels.
    """
    try:
        # Load the weather data from a CSV file
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

        # Preprocess the data
        # Assuming the CSV file has the following columns:
        # timestamp, temperature, humidity, pressure, wind_speed
        timestamps = data[:, 0]
        features = data[:, 1:]

        # Normalize the features
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        features = (features - mean) / std

        # Split the data into input sequences and corresponding labels
        seq_length = 24  # Length of the input sequence (e.g., 24 hours)
        num_samples = len(data) - seq_length

        # Create input sequences
        input_seqs = np.zeros((num_samples, seq_length, features.shape[1]))
        for i in range(num_samples):
            input_seqs[i] = features[i:i+seq_length]

        # Create corresponding labels (next temperature value)
        labels = features[seq_length:, 0]

        return input_seqs, labels

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None, None

    except Exception as e:
        print(f"Error: An unexpected error occurred during data loading: {str(e)}")
        return None, None


def prepare_time_series_data(data, labels, train_ratio=0.8):
    """
    Prepare the time series data for training and validation.

    Args:
        data (numpy.ndarray): Input sequences of weather data.
        labels (numpy.ndarray): Corresponding labels for the input sequences.
        train_ratio (float): Ratio of data to use for training (default: 0.8).

    Returns:
        tuple: A tuple containing the training and validation data and labels.
    """
    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, train_size=train_ratio, random_state=42)

    # Convert the data and labels to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)

    return train_data, val_data, train_labels, val_labels


def forecast_weather(model, data):
    """
    Forecast the weather using the trained model.

    Args:
        model (TimeSeriesTransformerModel): Trained time series transformer model.
        data (torch.Tensor): Input data for forecasting.

    Returns:
        numpy.ndarray: Predicted weather values.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    return predictions.detach().numpy()


def train_model(train_data, train_labels, val_data, val_labels, config=None, epochs=10, batch_size=32):
    """
    Train the time series transformer model.

    Args:
        train_data (torch.Tensor): Training input sequences.
        train_labels (torch.Tensor): Training labels.
        val_data (torch.Tensor): Validation input sequences.
        val_labels (torch.Tensor): Validation labels.
        config (TimeSeriesTransformerConfig): Configuration for the time series transformer model.
        epochs (int): Number of training epochs (default: 10).
        batch_size (int): Batch size for training (default: 32).

    Returns:
        TimeSeriesTransformerModel: Trained time series transformer model.
    """
    # Check if the input data and labels have the correct shape
    assert train_data.shape[0] == train_labels.shape[0], "Mismatch in the number of training samples"
    assert val_data.shape[0] == val_labels.shape[0], "Mismatch in the number of validation samples"

    # Create the time series transformer model
    if config is None:
        config = TimeSeriesTransformerConfig(
            input_size=train_data.shape[-1],
            num_heads=4,
            num_layers=2,
            output_size=1
        )
    model = TimeSeriesTransformerModel(config)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create data loaders for training and validation
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                val_loss += loss.item() * batch_data.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model


def evaluate_model(model, data, labels):
    """
    Evaluate the trained model on the given data.

    Args:
        model (TimeSeriesTransformerModel): Trained time series transformer model.
        data (torch.Tensor): Input data for evaluation.
        labels (torch.Tensor): Corresponding labels for evaluation.

    Returns:
        tuple: A tuple containing the evaluation metrics (MSE, MAE).
    """
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    predictions = predictions.detach().numpy()
    labels = labels.numpy()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)

    return mse, mae


def main():
    # Load and preprocess the weather data
    file_path = "weather_data.csv"
    data, labels = load_weather_data(file_path)

    if data is None or labels is None:
        print("Error: Failed to load weather data.")
        return

    # Prepare the time series data for training and validation
    train_data, val_data, train_labels, val_labels = prepare_time_series_data(data, labels)

    # Train the time series transformer model
    model = train_model(train_data, train_labels, val_data, val_labels, epochs=10, batch_size=32)

    # Evaluate the trained model
    mse, mae = evaluate_model(model, val_data, val_labels)
    print(f"Evaluation Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Forecast the weather using the trained model
    forecast_data = val_data[:10]  # Use a subset of validation data for forecasting
    forecasts = forecast_weather(model, forecast_data)
    print("Weather Forecasts:")
    print(forecasts)


if __name__ == "__main__":
    main()
