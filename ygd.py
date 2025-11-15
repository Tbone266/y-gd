# ygd.py
# Minimal implementation of Y-Gradient Descent (Y-GD)
# Baseline-relative proportional update with EMA baseline.

import torch

class YGD:
    def __init__(self, params, lr=0.1, gamma=0.05,
                 y_min=0.9, y_max=1.1):
        """
        params  - model parameters (iterable of tensors)
        lr      - step size
        gamma   - baseline smoothing rate
        y_min   - lower clamp for mediator
        y_max   - upper clamp for mediator
        """
        self.params = list(params)
        self.lr = lr
        self.y_min = y_min
        self.y_max = y_max
        self.gamma = gamma

        # Initialize baseline as copy of params
        self.baseline = [p.clone().detach() for p in self.params]

    @torch.no_grad()
    def step(self):
        """
        One Y-GD update step:
        1. compute proposed update
        2. compute ratio y
        3. clamp y
        4. update params relative to baseline
        5. update baseline
        """

        for p, z in zip(self.params, self.baseline):
            if p.grad is None:
                continue

            # Proposed raw GD step
            x = p.data
            g = p.grad.data
            x_proposed = x - self.lr * g

            # Ratio mediator
            num = (x_proposed - z).norm()
            den = (x - z).norm() + 1e-12  # avoid div by zero
            y = (num / den).clamp(self.y_min, self.y_max)

            # Y-GD update: baseline-relative interpolation
            p.data = z + y * (x - z)

            # Update baseline EMA
            z.mul_(1 - self.gamma).add_((1 - self.gamma) * x)



# demo_1d.py
import torch
import matplotlib.pyplot as plt
from ygd import YGD

# 1D quadratic f(x) = (x - 3)^2
def f(x):
    return (x - 3)**2

x = torch.tensor([8.0], requires_grad=True)
optimizer = YGD([x], lr=0.4, gamma=0.05, y_min=0.9, y_max=1.1)

traj = []
for t in range(50):
    x.grad = None
    y = f(x)
    y.backward()
    optimizer.step()
    traj.append(x.item())

plt.plot(traj, label="Y-GD trajectory")
plt.axhline(3, color="gray", linestyle="--", label="minimum")
plt.title("1D quadratic: Y-GD convergence")
plt.legend()
plt.savefig("ygd_1d.png")
plt.show()


# demo_2d.py
import torch
import matplotlib.pyplot as plt
from ygd import YGD

def f(x):
    # f(x, y) = x^2 + 10 y^2
    return x[0]**2 + 10 * x[1]**2

x = torch.tensor([4.0, -4.0], requires_grad=True)
optimizer = YGD([x], lr=0.3, gamma=0.05, y_min=0.8, y_max=1.2)

xs = []
ys = []

for t in range(80):
    x.grad = None
    loss = f(x)
    loss.backward()
    optimizer.step()
    xs.append(x[0].item())
    ys.append(x[1].item())

plt.plot(xs, ys, marker="o", markersize=2)
plt.title("2D anisotropic surface: Y-GD path")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig("ygd_2d.png")
plt.show()


torch
matplotlib
numpy


python demo_1d.py

python demo_2d.py





