import torch


class FedAvgOptimizer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def step(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()