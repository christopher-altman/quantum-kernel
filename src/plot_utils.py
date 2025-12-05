import matplotlib.pyplot as plt
import numpy as np

"""Plot utilities for datasets and expressivity metrics."""


def plot_dataset_scatter(X, y, title="Hybrid Quantum-Native Dataset"):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)
    plt.xlabel("x0 (normalized time-derived)")
    plt.ylabel("x1 (z0/zz mixture)")
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, quantum: bool = False, kernel_fn=None, X_train=None, title=None):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if quantum:
        from .quantum_kernels import compute_kernel_matrix
        assert kernel_fn is not None and X_train is not None
        K_grid = compute_kernel_matrix(kernel_fn, grid, X_train)
        Z = model.predict(K_grid)
    else:
        Z = model.predict(grid)

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title or "Decision Boundary")
    plt.tight_layout()
    plt.show()


def plot_metric_curve(values, x_axis, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.plot(x_axis, values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
