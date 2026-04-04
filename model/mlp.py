import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),

            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)