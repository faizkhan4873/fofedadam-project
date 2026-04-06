import numpy as np
from collections import defaultdict

def dirichlet_partition(X, y, num_clients=5, alpha=0.5, seed=42):
    

    np.random.seed(seed)

    num_classes = len(np.unique(y))
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]

    client_indices = defaultdict(list)

    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)


        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Convert proportions into splits
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for client_id, idx in enumerate(split_indices):
            client_indices[client_id].extend(idx)

    # Build final dataset
    client_data = {}
    for client_id in range(num_clients):
        idx = client_indices[client_id]
        client_data[client_id] = (X[idx], y[idx])

    return client_data