import torch
from optimizers.gl_fractional import GLFractionalGradient


class FOFedAdamW:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01,
                 alpha=0.7, K=10):

        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0  

        # Initialize moments
        self.m = {}
        self.v = {}

        
        self.gl_modules = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.m[name] = torch.zeros_like(param.data)
                self.v[name] = torch.zeros_like(param.data)

            
                self.gl_modules[name] = GLFractionalGradient(alpha=alpha, K=K)

    def step(self):
        self.t += 1

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data

            
            frac_grad = self.gl_modules[name].update(grad)

    
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * frac_grad

        
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (frac_grad ** 2)

            
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            
            update = m_hat / (torch.sqrt(v_hat) + self.eps)

            param.data = param.data - self.lr * update - self.lr * self.weight_decay * param.data

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()