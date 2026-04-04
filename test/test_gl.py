import torch
from optimizers.gl_fractional import GLFractionalGradient

gl = GLFractionalGradient(alpha=0.7, K=10)

for t in range(5):
    grad = torch.ones(3) * (t + 1)
    fg = gl.update(grad)
    print(f"Step {t}: {fg}")