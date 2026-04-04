import torch


class FedProxOptimizer:
    def __init__(self, model, global_model, lr=1e-3, mu=0.01):
        self.model = model
        self.global_model = global_model
        self.lr = lr
        self.mu = mu

    def step(self):
        for (name, param), (_, global_param) in zip(
                self.model.named_parameters(),
                self.global_model.named_parameters()):

            if param.grad is None:
                continue

            prox_term = self.mu * (param.data - global_param.data)
            param.data -= self.lr * (param.grad.data + prox_term)

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()