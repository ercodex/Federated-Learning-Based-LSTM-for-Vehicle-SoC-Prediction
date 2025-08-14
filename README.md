# Federated Learning-Based LSTM for Vehicle SoC Prediction

A **Federated Learning** approach for predicting the **State of Charge (SoC)** of electric vehicles using an **LSTM** model.  
The system is trained on a **heterogeneous dataset** distributed across **5 clients** and evaluated on a separate **test set**.  
Both **homogeneous** and **heterogeneous** client data distributions are investigated. 

It is easier to apply homogeneous techniques since we can apply some changes on datasets. For example, adding missing columns
between clients and filling them with 0. Although, our ultimate goal was to get a better performance from heterogeneous techniques. 
We didn't deal with data this time and feed the model with what we have.

---

## Features
- **Federated Learning** with 5 clients + 1 global test set
- **LSTM architecture** for time-series based SoC prediction
- Experiments with **homogeneous vs. heterogeneous** data distributions
- MAE, MAPE, MSE, and R^2 Evaluation metrics for performance comparison

---

## Dataset
The dataset contains electric vehicle telemetry data.
Two distribution strategies are applied:

1. **Homogeneous** – Similar data distribution across all clients. Each client has the same 5 features in this one.
2. **Heterogeneous** – Non-IID distribution simulating real-world scenarios.

---

## Model Architecture
- **Input Layer**: Sequential EV telemetry data
- **LSTM Layers**: For temporal dependency learning
- **Dense Layer**: For regression output
- **Output**: Predicted State of Charge (%)

---

## Installation
```bash
# Clone repository
git clone https://github.com/ercodex/Federated-Learning-Based-LSTM-for-Vehicle-SoC-Prediction.git
cd federated-lstm-vehicle-soc

# Install requirements
pip install -r requirements.txt
