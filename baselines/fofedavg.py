from optimizers.gl_fractional import GLFractionalGradient


class FOFedAvgOptimizer:
    def __init__(self, model, lr=1e-3, alpha=0.7, K=10):
        self.model = model
        self.lr = lr

        self.gl_modules = {
            name: GLFractionalGradient(alpha, K)
            for name, _ in model.named_parameters()
        }

    def step(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            frac_grad = self.gl_modules[name].update(param.grad.data)
            param.data -= self.lr * frac_grad

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()