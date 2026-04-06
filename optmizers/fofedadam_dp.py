import torch
from optimizers.fofedadamw import FOFedAdamW


class FOFedAdamW_DP(FOFedAdamW):
    def __init__(self, model, clip_norm=1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.clip_norm = clip_norm

    def clip_gradients(self):
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2

        total_norm = total_norm ** 0.5

        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data *= clip_coef

    def step(self):
        
        self.clip_gradients()

        super().step()