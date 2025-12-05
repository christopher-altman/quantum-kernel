"""Quantum Kernel Expressivity – core package."""

from .dataset_c4 import generate_hybrid_dataset
from .quantum_kernels import make_kernel_circuit, make_noisy_kernel_circuit, compute_kernel_matrix
from .analysis_utils import kernel_spectral_entropy, kernel_matrix_rank, kernel_spectrum, numerical_gradient
from .vqc_classifier import make_vqc_classifier, train_vqc, vqc_predict
from .classical_baselines import classical_rbf_svm, classical_poly_svm, classical_logistic
