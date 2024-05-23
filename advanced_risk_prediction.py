# advanced_risk_prediction.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data(file_path):
    """
    Load and preprocess the risk assessment data.

    Args:
        file_path (str): Path to the CSV file containing risk assessment data.

    Returns:
        tuple: A tuple containing the preprocessed data and labels.
    """
    try:
        # Load the data from a CSV file
        data = pd.read_csv(file_path)

        # Preprocess the data
        # Assuming the CSV file has the following columns:
        # 'feature1', 'feature2', ..., 'risk_label'
        features = data.drop('risk_label', axis=1)
        labels = data['risk_label']

        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        return scaled_features, labels

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None, None

    except KeyError as e:
        print(f"Error: Required column not found in the data: {str(e)}")
        return None, None

    except Exception as e:
        print(f"Error: An unexpected error occurred during data loading: {str(e)}")
        return None, None

def create_model(input_shape):
    """
    Create a deep learning model for risk prediction.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: A compiled deep learning model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=5):
    """
    Train the risk prediction model.

    Args:
        model (Sequential): The risk prediction model.
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        epochs (int): Number of training epochs (default: 50).
        batch_size (int): Batch size for training (default: 32).
        patience (int): Number of epochs with no improvement after which training will be stopped (default: 5).

    Returns:
        tuple: A tuple containing the trained model and training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained risk prediction model.

    Args:
        model (Sequential): The trained risk prediction model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def predict_risk(model, data):
    """
    Predict the risk using the trained model.

    Args:
        model (Sequential): The trained risk prediction model.
        data (numpy.ndarray): Input data for risk prediction.

    Returns:
        numpy.ndarray: Predicted risk probabilities.
    """
    risk_probabilities = model.predict(data)
    return risk_probabilities

def main():
    # Load and preprocess the risk assessment data
    file_path = 'risk_assessment_data.csv'
    data, labels = load_data(file_path)

    if data is None or labels is None:
        print("Error: Failed to load data.")
        return

    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create the risk prediction model
    input_shape = (X_train.shape[1],)
    model = create_model(input_shape)

    # Train the model
    model, history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the trained model
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Predict risk for new data
    new_data = np.array([[0.5, 0.7, 0.2, 0.9]])  # Example new data
    risk_probabilities = predict_risk(model, new_data)
    print("Risk Probabilities:")
    print(risk_probabilities)

if __name__ == "__main__":
    main()
