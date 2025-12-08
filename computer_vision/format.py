import numpy as np
import matplotlib.pyplot as plt

p = np.array([0, 0])
grid = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
x, y = grid
X = np.c_[x.ravel(), y.ravel()]

L1 = np.abs(X - p).sum(axis=1).reshape(x.shape)
L2 = np.linalg.norm(X - p, axis=1).reshape(x.shape)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.contourf(x, y, L1, levels=[0, 1], colors=["lightblue"], alpha=0.7)
plt.contour(x, y, L1, levels=[1], colors="blue", linewidths=2)
plt.scatter(0, 0, c="red", s=100, zorder=5)
plt.title("L1 = Diamond")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.contourf(x, y, L2, levels=[0, 1], colors=["lightcoral"], alpha=0.7)
plt.contour(x, y, L2, levels=[1], colors="red", linewidths=2)
plt.scatter(0, 0, c="red", s=100, zorder=5)
plt.title("L2 = Circle")
plt.axis("equal")

plt.tight_layout()
plt.show()
