import numpy as np
import pennylane as qml
from .hamiltonian_evolution import make_evolution_qnode

"""Hybrid quantum-native dataset (C4) using T2 Hamiltonian evolution."""

DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 42
T_MIN, T_MAX = 0.0, 2 * np.pi


def _sample_state_params(rng: np.random.Generator) -> np.ndarray:
    theta1 = rng.uniform(0, np.pi)
    phi1 = rng.uniform(0, 2 * np.pi)
    theta2 = rng.uniform(0, np.pi)
    phi2 = rng.uniform(0, 2 * np.pi)
    return np.array([theta1, phi1, theta2, phi2], dtype=float)


def generate_hybrid_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
):
    """
    Generate (X, y, meta) with:
        X: (n_samples, 2) float32
        y: (n_samples,) int labels {0,1}
    """
    rng = np.random.default_rng(seed)
    dev = qml.device("default.qubit", wires=2)
    evo = make_evolution_qnode(device=dev)

    X = np.zeros((n_samples, 2), dtype=float)
    y = np.zeros((n_samples,), dtype=int)

    t_vals = np.zeros(n_samples, dtype=float)
    z0_vals = np.zeros(n_samples, dtype=float)
    zz_vals = np.zeros(n_samples, dtype=float)
    params_all = np.zeros((n_samples, 4), dtype=float)

    for i in range(n_samples):
        params = _sample_state_params(rng)
        t = rng.uniform(T_MIN, T_MAX)

        # QNode returns PL tensors; cast to float
        z0, zz = evo(params, t)
        z0 = float(z0)
        zz = float(zz)

        x0 = t / (2 * np.pi)
        x1 = 0.5 * z0 + 0.5 * zz

        X[i, 0] = x0
        X[i, 1] = x1

        signal = z0 + 0.7 * zz + 0.3 * np.cos(t)
        y[i] = 1 if signal > 0 else 0

        t_vals[i] = t
        z0_vals[i] = z0
        zz_vals[i] = zz
        params_all[i] = params

    # Standardize features for sklearn
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std

    # Ensure pure NumPy dtypes for sklearn
    X_scaled = np.asarray(X_scaled, dtype=float)
    y = np.asarray(y, dtype=int)

    meta = {
        "t": t_vals,
        "z0": z0_vals,
        "zz": zz_vals,
        "params": params_all,
        "X_mean": X_mean,
        "X_std": X_std,
    }

    return X_scaled, y, meta