import pennylane as qml
from pennylane import numpy as np

"""Implements T2 entangling Hamiltonian evolution and an alternate Hamiltonian."""

N_DATA_QUBITS = 2

# T2 Hamiltonian: H = Z⊗Z + X⊗I
coeffs = [1.0, 1.0]
ops = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)]
H_T2 = qml.Hamiltonian(coeffs, ops)

# Alternate Hamiltonian for contrast
coeffs_alt = [1.0, 0.5]
ops_alt = [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
H_alt = qml.Hamiltonian(coeffs_alt, ops_alt)


def _prepare_product_state(params):
    theta1, phi1, theta2, phi2 = params
    qml.RY(theta1, wires=0)
    qml.RZ(phi1, wires=0)
    qml.RY(theta2, wires=1)
    qml.RZ(phi2, wires=1)


def make_evolution_qnode(device=None):
    dev = device or qml.device("default.qubit", wires=N_DATA_QUBITS)

    @qml.qnode(dev)
    def evolved_expectations(params, t):
        """Return (⟨Z0⟩, ⟨Z0 Z1⟩) after T2 evolution."""
        _prepare_product_state(params)
        qml.ApproxTimeEvolution(H_T2, t, 2)
        z0 = qml.expval(qml.PauliZ(0))
        zz = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        return z0, zz

    return evolved_expectations


def sample_temporal_trajectory(params, t_values, device=None):
    evo = make_evolution_qnode(device=device)
    z0_list, zz_list = [], []
    for t in t_values:
        z0, zz = evo(params, t)
        z0_list.append(z0)
        zz_list.append(zz)
    return np.array(z0_list), np.array(zz_list)


def make_alternate_evolution_qnode(device=None):
    dev = device or qml.device("default.qubit", wires=N_DATA_QUBITS)

    @qml.qnode(dev)
    def evolved(params, t):
        _prepare_product_state(params)
        qml.ApproxTimeEvolution(H_alt, t, 2)
        z0 = qml.expval(qml.PauliZ(0))
        z1 = qml.expval(qml.PauliZ(1))
        return z0, z1

    return evolved
