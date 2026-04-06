import torch
import torch.nn as nn
from optimizers.fofedadamw import FOFedAdamW


model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

optimizer = FOFedAdamW(model)


X = torch.randn(32, 10)
y = torch.randn(32, 1)

criterion = nn.MSELoss()

for epoch in range(5):
    optimizer.zero_grad()

    output = model(X)
    loss = criterion(output, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")