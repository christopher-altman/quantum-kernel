import sys
import os

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(NOTEBOOK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset_c4 import generate_hybrid_dataset
from src.plot_utils import plot_dataset_scatter, plot_decision_boundary, plot_metric_curve
from src.classical_baselines import classical_rbf_svm, classical_poly_svm, classical_logistic
from src.quantum_kernels import make_kernel_circuit, make_noisy_kernel_circuit, compute_kernel_matrix
from src.analysis_utils import kernel_spectral_entropy, kernel_matrix_rank, numerical_gradient
from src.vqc_classifier import make_vqc_classifier, train_vqc, vqc_predict
