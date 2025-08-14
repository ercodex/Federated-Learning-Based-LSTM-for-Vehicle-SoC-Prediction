""" This code makes the visualization of our mode """

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

TEST_DATA_PATH = r"C:\Users\Erncl\Desktop\ercodex\Projects_Eren_Coding\ICT Summer School\Federated Learning Mini Project\Test_dataset.csv"

# The exact LSTM architecture with client codes
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=200, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def create_sequences(input_data, target_data, sequence_length):
    sequences, labels = [], []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i:i + sequence_length])
        labels.append(target_data[i + sequence_length])
    return np.array(sequences), np.array(labels)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Main function
if __name__ == "__main__":
    
    # Check files
    required_files = ["global_model.pt", "scaler_x.pkl", "scaler_y.pkl"]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Couldn't find '{f}' files. Please be sure to run previous scripts.")

    # Load the model and scalers
    model_params = torch.load("global_model.pt", weights_only=False) # Gives an UnplickingError if 'weights_only' left default, True.
    scaler_x = joblib.load("scaler_x.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    
    # Load test file
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError("Please provide a test file")

    # All possible solutions
    all_possible_features = ["PACK_V_SUM_OF_CELLS", "PACK_I_HALL", "CELL_T_MAX_VAL", "vehicle_speed", "PACK_Q_SOC_INTERNAL_t-1"]
    
    # Process test file
    test_df['PACK_Q_SOC_INTERNAL_t-1'] = test_df['PACK_Q_SOC_INTERNAL'].shift(1)
    test_df = test_df.dropna().reset_index(drop=True)
    for col in all_possible_features:
        if col not in test_df.columns:
            test_df[col] = 0

    X_test_raw = test_df[all_possible_features]
    y_test_raw = test_df[['PACK_Q_SOC_INTERNAL']]
    
    # Scale data
    X_test_scaled = scaler_x.transform(X_test_raw)
    
    # Create array for test
    X_test_seq, _ = create_sequences(X_test_scaled, y_test_raw.values, 10)
    X_test_tensor = torch.from_numpy(X_test_seq).float()

    model = LSTMModel(input_size=5)
    
    params_dict = zip(model.state_dict().keys(), model_params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Guess
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor)
    
    # Re-scale the guesses
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())
    
    # Re-scale the y values too
    y_test_actual = y_test_raw.iloc[10:].values

    # Calculate error metrics
    mae = mean_absolute_error(y_test_actual, predictions)
    mape = mean_absolute_percentage_error(y_test_actual, predictions)
    mse = mean_squared_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}") 
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-Squared (R²): {r2:.4f}")
print("-------------------------------------\n")

# G17: Plot the actual and predicted results
plt.figure(figsize=(15, 7))
plt.plot(y_test_actual, label="Actual SoC Values", color="blue", linewidth=2)
plt.plot(predictions, label="Predicted SoC Values", color="red", linestyle="--")
plt.title("SoC Prediction: Actual vs. Predicted Results")
plt.xlabel("Time Steps (Test Data)")
plt.ylabel("State of Charge (SoC) [%]")
plt.legend()
plt.grid(True)
plt.savefig("output_graph(s).png")
print("Prediction graph saved as 'output_graph(s).png'.")
plt.show()