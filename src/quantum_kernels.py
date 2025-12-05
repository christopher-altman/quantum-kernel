import pennylane as qml
from pennylane import numpy as np

"""Quantum kernel constructors and utilities."""

DEFAULT_N_QUBITS = 4


def make_kernel_circuit(n_qubits: int = DEFAULT_N_QUBITS, omega: float = 8.0):
    """Return a QNode computing |⟨ψ(x1)|ψ(x2)⟩|² via a |0…0⟩ projector."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        wires = list(range(n_qubits))

        # Encode x1
        from .feature_maps import high_frequency_feature_map
        high_frequency_feature_map(x1, wires, omega=omega)

        # Uncompute x2
        qml.adjoint(high_frequency_feature_map)(x2, wires, omega=omega)

        # Probability of returning to |0…0⟩ is the squared overlap
        return qml.expval(qml.Projector([0] * n_qubits, wires=wires))

    return kernel_circuit


def make_noisy_kernel_circuit(
    n_qubits: int = DEFAULT_N_QUBITS,
    omega: float = 8.0,
    noise_level: float = 0.01,
):
    """Noisy kernel circuit with depolarizing channels applied after encoding."""
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def noisy_kernel_circuit(x1, x2):
        wires = list(range(n_qubits))

        from .feature_maps import high_frequency_feature_map
        high_frequency_feature_map(x1, wires, omega=omega)

        # Apply local depolarizing noise
        for w in wires:
            qml.DepolarizingChannel(noise_level, wires=w)

        # Uncompute x2
        qml.adjoint(high_frequency_feature_map)(x2, wires, omega=omega)

        return qml.expval(qml.Projector([0] * n_qubits, wires=wires))

    return noisy_kernel_circuit


def compute_kernel_matrix(kernel_fn, X1, X2, batch_size: int | None = None):
    """Compute Gram matrix K_ij = kernel_fn(X1[i], X2[j])."""
    X1 = np.array(X1)
    X2 = np.array(X2)
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))

    if batch_size is None or batch_size >= n1:
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_fn(X1[i], X2[j])
        return K

    for start in range(0, n1, batch_size):
        end = min(start + batch_size, n1)
        for i in range(start, end):
            for j in range(n2):
                K[i, j] = kernel_fn(X1[i], X2[j])
    return K