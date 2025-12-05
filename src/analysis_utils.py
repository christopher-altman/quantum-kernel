import numpy as np
from scipy.linalg import eigh

"""Analysis utilities for kernel spectra and expressivity."""


def kernel_spectral_entropy(K: np.ndarray) -> float:
    vals, _ = eigh(K)
    vals = np.clip(vals, 1e-12, None)
    p = vals / np.sum(vals)
    return float(-np.sum(p * np.log(p)))


def kernel_matrix_rank(K: np.ndarray, tol: float = 1e-8) -> int:
    vals = np.linalg.eigvalsh(K)
    return int(np.sum(vals > tol))


def kernel_spectrum(K: np.ndarray):
    vals = np.linalg.eigvalsh(K)
    return np.sort(vals)[::-1]


def numerical_gradient(values, x_axis):
    values = np.array(values)
    x_axis = np.array(x_axis)
    return np.gradient(values, x_axis)
