import numpy as np
from data.dirichlet_partition import dirichlet_partition

# dummy dataset
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

clients = dirichlet_partition(X, y, num_clients=5, alpha=0.5)

for i, (x_c, y_c) in clients.items():
    print(f"Client {i}: {len(x_c)} samples, class ratio = {np.mean(y_c):.2f}")