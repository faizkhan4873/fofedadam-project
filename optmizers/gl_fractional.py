import torch
import math


class GLFractionalGradient:
    def __init__(self, alpha=0.7, K=10):
     
        self.alpha = alpha
        self.K = K
        self.history = []

        
        self.coeffs = self._compute_coeffs()

    def _compute_coeffs(self):
        coeffs = []
        for k in range(self.K + 1):
            coeff = ((-1) ** k) * self._binomial(self.alpha, k)
            coeffs.append(coeff)
        return coeffs

    def _binomial(self, alpha, k):
        if k == 0:
            return 1.0
        num = 1.0
        for i in range(k):
            num *= (alpha - i)
        return num / math.factorial(k)

    def update(self, grad):
        
        # Store history
        self.history.insert(0, grad.clone())
        if len(self.history) > self.K:
            self.history.pop()

        # Compute GL sum
        frac_grad = torch.zeros_like(grad)

        for k in range(len(self.history)):
            frac_grad += self.coeffs[k] * self.history[k]

        return frac_grad