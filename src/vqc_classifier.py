import pennylane as qml
from pennylane import numpy as np
from .feature_maps import high_frequency_feature_map

"""Variational Quantum Classifier (VQC) with StronglyEntanglingLayers."""

DEFAULT_N_QUBITS = 4


def make_vqc_classifier(n_qubits: int = DEFAULT_N_QUBITS, n_layers: int = 2):
    """Return a VQC QNode with data re-uploading followed by SEL variational block."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        wires = list(range(n_qubits))

        # Single data embedding pass
        high_frequency_feature_map(x, wires, omega=4.0)

        # Variational block: all layers applied once
        qml.StronglyEntanglingLayers(weights, wires=wires)

        return qml.expval(qml.PauliZ(0))

    # Attach metadata so training code can infer layer count
    circuit.n_qubits = n_qubits
    circuit.n_layers = n_layers
    return circuit


def _init_weights(n_qubits: int, n_layers: int, rng=None):
    """Initialize SEL weights with small random values."""
    if rng is None:
        rng = np.random.default_rng(42)
    return 0.01 * rng.standard_normal(size=(n_layers, n_qubits, 3))


def train_vqc(model, X, y, n_steps: int = 100, lr: float = 0.05, verbose: bool = True):
    """
    Train VQC with mean-squared error on probabilities.

    Args:
        model: QNode from make_vqc_classifier
        X: features (n_samples, d)
        y: labels in {0,1}
    """
    n_qubits = getattr(model, "n_qubits", 4)
    n_layers = getattr(model, "n_layers", 2)
    weights = _init_weights(n_qubits, n_layers=n_layers)

    opt = qml.GradientDescentOptimizer(stepsize=lr)

    def loss_fn(w):
        preds = []
        for xi, yi in zip(X, y):
            z = model(w, xi)
            p = (1.0 + z) / 2.0
            preds.append(p)
        preds = np.array(preds)
        return np.mean((preds - y) ** 2)

    losses = []
    for step in range(n_steps):
        weights, current_loss = opt.step_and_cost(loss_fn, weights)
        losses.append(current_loss)
        if verbose and (step % max(1, n_steps // 10) == 0):
            print(f"[VQC] Step {step:3d} | Loss = {current_loss:.4f}")
    return weights, losses


def vqc_predict(model, weights, X):
    """Predict hard labels {0,1} from trained VQC."""
    preds = []
    for x in X:
        z = model(weights, x)
        p = (1.0 + z) / 2.0
        preds.append(1 if p > 0.5 else 0)
    return np.array(preds, dtype=int)