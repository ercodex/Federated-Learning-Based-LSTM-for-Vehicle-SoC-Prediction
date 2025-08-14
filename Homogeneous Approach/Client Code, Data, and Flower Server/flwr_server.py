""" This code generally 'manages' the Federated Learning process """

import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays

# We just add the 'save the model' operation. This why it is a 'special' strategy.
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        # Take every model weight that comes from each client and take the standard weighted avg.
        # Then assign it to 'aggregated_parameters' variable.
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # If rounds are done and aggregated_parameters exists, save the model
        if aggregated_parameters is not None and server_round == 10: # 10 rounds
            print(f"Last round is ({server_round}) completed. Saving the global model...")
            
            # Turn parameters into NumPy arrays
            np_arrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Save the model in .pt format
            torch.save(np_arrays, "global_model.pt")
            print("Model 'global_model.pt' saved successfully.")

        return aggregated_parameters, aggregated_metrics

# Configure the strategy with server side parameters provided in the document.
strategy = SaveModelStrategy(
    fraction_fit=1.0,           # Involve 100% of each client in every round.
    fraction_evaluate=0.5,      # Use 50% of clients for evaluation.
    min_fit_clients=5,          # 5 clients are required.
    min_evaluate_clients=5,     # 5 clients are required for evaluation.
    min_available_clients=5,    # Wait for all 5 clients to connect before starting.
)

# Start the server
print("Federated Learning server is running...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10), # 10 rounds of training.
    strategy=strategy,
)