import torch
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from plot import plot_auc
from evaluate import evaluate_model

from models.mlp import MLP
from optimizers.fofedadamw_dp import FOFedAdamW_DP
from server.aggregator import fedavg_aggregate_dp

def client_update(global_model, X, y, epochs=5,optimizer_type="fofedadamw", batch_size=32, dp=False):
    model = deepcopy(global_model)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if optimizer_type == "fedavg":
    from baselines.fedavg import FedAvgOptimizer
    optimizer = FedAvgOptimizer(model)

    elif optimizer_type == "fedprox":
    from baselines.fedprox import FedProxOptimizer
    optimizer = FedProxOptimizer(model, global_model)

    elif optimizer_type == "fedadam":
    from baselines.fedadam import FedAdam
    optimizer = FedAdam(model)

    elif optimizer_type == "fofedavg":
    from baselines.fofedavg import FOFedAvgOptimizer
    optimizer = FOFedAvgOptimizer(model)

    elif optimizer_type == "fofedadamw":
    optimizer = FOFedAdamW(model)

    elif optimizer_type == "fofedadamw_dp":
    optimizer = FOFedAdamW_DP(model, clip_norm=1.0)
    
    criterion = torch.nn.BCELoss()

    model.train()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()

            preds = model(xb)
            loss = criterion(preds, yb)

            loss.backward()
            optimizer.step()

    return model, len(dataset)

  

def federated_training(client_data, input_dim, rounds=10, dp=False):
    global_model = MLP(input_dim)

    history = {"auc": []}

    X_all = np.vstack([client_data[c][0] for c in client_data])
    y_all = np.hstack([client_data[c][1] for c in client_data])

    for r in range(rounds):
        client_models = []
        client_sizes = []

        print(f"\n--- Round {r} ---")

        for client_id, (X, y) in client_data.items():
            model, size = client_update(global_model, X, y, dp=dp)
            client_models.append(model)
            client_sizes.append(size)

        # 🔥 Choose aggregation
        if dp:
            global_model = fedavg_aggregate_dp(global_model, client_models, client_sizes)
        else:
            global_model = fedavg_aggregate(global_model, client_models, client_sizes)

        acc, auc, f1 = evaluate_model(global_model, X_all, y_all)
        history["auc"].append(auc)

        print(f"AUC: {auc:.4f}")

    return global_model, history
  
  
if __name__ == "__main__":
    import numpy as np
    from data.dirichlet_partition import dirichlet_partition

    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    clients = dirichlet_partition(X, y, num_clients=5, alpha=0.5)
    
    methods = ["fedavg", "fedprox", "fedadam", "fofedavg", "fofedadamw"]

results = {}

for method in methods:
    print(f"\nRunning {method}")

    _, history = federated_training(
        clients,
        input_dim=10,
        rounds=10,
        dp=(method == "fofedadamw_dp")
    )

    results[method] = history["auc"]

    model, history = federated_training(clients, input_dim=10, rounds=10, dp=True)

    plot_auc(history)