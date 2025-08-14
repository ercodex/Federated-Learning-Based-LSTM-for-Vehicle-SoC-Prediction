""" 
Client codes are responsible for handling the data, creating the model, training the model,
and communicate with server. These files are the core of the project.

Eren Çil
Süleyman Efe Polat
"""

# Import libraries
import flwr as fl # Federated Learning framework

import torch # Deep Learning Framework 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np # Math operations
import pandas as pd

import joblib # For creating pickle file

import os # Operating system

""" Client 1 Configurations """
CLIENT_ID = 1 # Client ID

FEATURES = [ # Features of client 
    "PACK_V_SUM_OF_CELLS",
    "PACK_I_HALL",
    "PACK_Q_SOC_INTERNAL_t-1"
]   
NUM_FEATURES = 5 # Client 1 input size
DATA_PATH = f"Client{CLIENT_ID}.csv" # Client training file path
# -----------------------------

class LSTMModel(nn.Module):
    """ Model Class """
    # Constructor function
    def __init__(self, input_size, hidden_layer_size=200, output_size=1): # We are guessing 1 feature.
        super().__init__()
        
        self.hidden_layer_size = hidden_layer_size # Hidden layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True) # LSTM Layer
        self.linear = nn.Linear(hidden_layer_size, output_size) # Output layer

    # Forward-prop function
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq) # Data goes into LSTM and creates an output.
        predictions = self.linear(lstm_out[:, -1, :]) # Only take the output of the last step. Then give it to linear layer to create an output
        
        return predictions
# -----------------------------

def create_sequences(input_data, target_data, sequence_length):
    """ Sliding window function. Creates training sequences for LSTM Model """
    
    # Create two empty arrays
    sequences, labels = [], []
    
    # Start loop
    for i in range(len(input_data) - sequence_length): # Sequence length is subtracted to not overflow from array
        
        # Save it
        seq = input_data[i:i + sequence_length] # The window that contains train data
        label = target_data[i + sequence_length] # The target variable
        
        # Append the values to the array
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)
# -----------------------------

def load_data(sequence_length=10):
    """
    Loads data, transforms it with global scalers, and prepares for DataLoader. This function 
    always returns 5-featured-data.
    """

    # If can't find scalers, raise an error
    if not os.path.exists('scaler_x.pkl') or not os.path.exists('scaler_y.pkl'):
        raise FileNotFoundError("Couldn't find global scalers 'scaler_x.pkl' or 'scaler_y.pkl'.")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # This is where the magic happens. We also feed the model with the previous SoC value.
    # This way, it can predict better.
    df['PACK_Q_SOC_INTERNAL_t-1'] = df['PACK_Q_SOC_INTERNAL'].shift(1) # !!!
    df = df.dropna().reset_index(drop=True) # We clean the NaN value in the first index.
    
    # This part works different for each client. We standardize every one of them.
    all_possible_features = ["PACK_V_SUM_OF_CELLS", "PACK_I_HALL", "CELL_T_MAX_VAL", "vehicle_speed", "PACK_Q_SOC_INTERNAL_t-1"]
    for col in all_possible_features: 
        if col not in df.columns: # If client doesn't have the specific column.
            df[col] = 0 # Create and fill it with 0.

    # Load pre-trained global scalers
    scaler_x = joblib.load('scaler_x.pkl') 
    scaler_y = joblib.load('scaler_y.pkl')

    # In 'create_scaler.py', we applied 'fit()' to data.
    # Now, it is time to 'transform()'.
    X_scaled_all_features = scaler_x.transform(df[all_possible_features])
    y_scaled = scaler_y.transform(df[['PACK_Q_SOC_INTERNAL']])

    # We use 'create_sequences()' function here. We always pass the 5-featured-data to it.
    X_seq, y_seq = create_sequences(X_scaled_all_features, y_scaled, sequence_length)

    # Turn data into tensors.
    X_tensor = torch.from_numpy(X_seq).float()
    y_tensor = torch.from_numpy(y_seq).float()
    
    dataset = TensorDataset(X_tensor, y_tensor) # !!!
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Randomised groups of 32
    
    # Return DataLoader instance and the length of X_tensor. !!!
    return train_loader, len(X_tensor)
# -----------------------------

def train(model, train_loader, epochs=10):
    """ Train the model """
    criterion = nn.MSELoss() # Mean-Square Error Loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam optimizer. The Classic.

    model.train() # This line tells PyTorch that out model is now in 'train mode'.

    # Train loop
    for epoch in range(epochs):

        for seq, labels in train_loader:

            optimizer.zero_grad()            # Clear the gradients from last step.
            y_pred = model(seq)              # Forward-pass
            loss = criterion(y_pred, labels) # Calculate the loss
            loss.backward()                  # Back-prop
            optimizer.step()                 # Update model weights
# -----------------------------

# Flower NumPyClient
# We tell this class to follow 'NumPyClient Protocol'. It specifies that this object is going to transfer model weights in NumPy arrays format.
# PyTorch uses tensors, but it seems like Flower uses NumPy. So we need to make some conversions.
class SocClient(fl.client.NumPyClient): 
    """ 
    This class serves as a bridge between PyTorch and Flower frameworks. It also provides
    the connection between the main server and this specific client. They are always in touch
    during the training phase.

    P.S.
    We as a team, have the least information and experience about this part. So, we got A LOT of
    help from LLMs.
    """

    # Constructor function
    def __init__(self, model, train_loader, num_examples):
        
        # Object arguments
        self.model = model
        self.train_loader = train_loader
        self.num_examples = num_examples # Main server will ask for this

    # Main server will get current data by using this function
    def get_parameters(self, config):

        # This PyTorch command gives all the trainable parameters (weights and bias) of model as a dictionary.
        # Don't forget to send the data to CPU back, if it is trained on a GPU, it would be an issue.
        # Convert to NumPy array.
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()] # We don't use the key of dictionary. So, pass it with '_'.
    
    # Every 'get' has its 'set'. I think we all know what this functions does.
    def set_parameters(self, parameters):

        # Try to match features with clients who has different number of parameters  
        own_state = self.model.state_dict()
        
        for (name, own_param), server_param in zip(own_state.items(), parameters):
            if own_param.shape == torch.from_numpy(server_param).shape:
                own_state[name] = torch.from_numpy(server_param)
        
        self.model.load_state_dict(own_state, strict=False)

    # Now it is time to fit 
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=10)
        return self.get_parameters(config={}), self.num_examples, {}

    # Report the performance 
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.MSELoss()
        loss = 0.0
        total = 0
        self.model.eval() # Now model is on 'evaluation' mode
        with torch.no_grad():
            for seq, labels in self.train_loader:
                outputs = self.model(seq)
                
                loss += criterion(outputs, labels).item() * len(labels)
                total += len(labels)
    
        # Calculate avg loss
        avg_loss = loss / total
        return avg_loss, self.num_examples, {"loss": avg_loss}

# Main Function
if __name__ == "__main__":

    print(f"Starting Client {CLIENT_ID}!")
    
    train_loader, num_examples = load_data()
    model = LSTMModel(input_size=NUM_FEATURES) # Client-Unique Model

    client = SocClient(model, train_loader, num_examples)
    
    # Connect the server and start process
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    
    print(f"Client {CLIENT_ID} completed it's training.")