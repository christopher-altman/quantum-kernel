"""Generate key figures into diagrams/ and numeric results into results/."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from src.dataset_c4 import generate_hybrid_dataset
from src.quantum_kernels import make_kernel_circuit, make_noisy_kernel_circuit, compute_kernel_matrix
from src.analysis_utils import kernel_spectral_entropy, kernel_matrix_rank

ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = ROOT / "diagrams"
RESULTS_DIR = ROOT / "results"
DIAGRAMS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_dataset_scatter():
    X, y, meta = generate_hybrid_dataset(n_samples=500, seed=RANDOM_SEED)
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title("Hybrid Quantum-Native Dataset (C4 + T2)")
    plt.xlabel("Feature 1 (time-normalized)")
    plt.ylabel("Feature 2 (z0/zz mixed signal)")
    plt.tight_layout()
    out = DIAGRAMS_DIR / "dataset_scatter.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[OK] Saved {out}")


def generate_expressivity_sweep_figures():
    X, y, meta = generate_hybrid_dataset(n_samples=500, seed=RANDOM_SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    omega_values = np.linspace(1.0, 20.0, 10)
    accs_ideal, accs_noisy = [], []
    ranks_ideal, ents_ideal, ents_noisy = [], [], []

    for omega in omega_values:
        print(f"[Sweep] ω = {omega:.2f}")
        kernel_fn = make_kernel_circuit(n_qubits=4, omega=omega)
        K_train = compute_kernel_matrix(kernel_fn, X_train, X_train)
        K_test = compute_kernel_matrix(kernel_fn, X_test, X_train)

        clf = SVC(kernel="precomputed")
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        accs_ideal.append(accuracy_score(y_test, y_pred))

        ranks_ideal.append(kernel_matrix_rank(K_train))
        ents_ideal.append(kernel_spectral_entropy(K_train))

        noisy_kernel_fn = make_noisy_kernel_circuit(n_qubits=4, omega=omega, noise_level=0.01)
        K_train_n = compute_kernel_matrix(noisy_kernel_fn, X_train, X_train)
        K_test_n = compute_kernel_matrix(noisy_kernel_fn, X_test, X_train)

        clf_n = SVC(kernel="precomputed")
        clf_n.fit(K_train_n, y_train)
        y_pred_n = clf_n.predict(K_test_n)
        accs_noisy.append(accuracy_score(y_test, y_pred_n))
        ents_noisy.append(kernel_spectral_entropy(K_train_n))

    data = {
        "omega_values": omega_values.tolist(),
        "accs_ideal": [float(a) for a in accs_ideal],
        "accs_noisy": [float(a) for a in accs_noisy],
        "ranks_ideal": [int(r) for r in ranks_ideal],
        "ents_ideal": [float(e) for e in ents_ideal],
        "ents_noisy": [float(e) for e in ents_noisy],
    }
    with open(RESULTS_DIR / "expressivity_sweep.json", "w") as f:
        json.dump(data, f, indent=2)

    plt.figure(figsize=(7, 5))
    plt.plot(omega_values, accs_ideal, marker="o")
    plt.xlabel("ω (encoding frequency)")
    plt.ylabel("Accuracy")
    plt.title("Quantum Kernel Accuracy vs ω (Ideal)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "accuracy_vs_omega_ideal.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(omega_values, accs_noisy, marker="o")
    plt.xlabel("ω (encoding frequency)")
    plt.ylabel("Accuracy")
    plt.title("Quantum Kernel Accuracy vs ω (Noisy, p=0.01)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "accuracy_vs_omega_noisy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(omega_values, ranks_ideal, marker="o")
    plt.xlabel("ω (encoding frequency)")
    plt.ylabel("Rank(K_train)")
    plt.title("Kernel Matrix Rank vs ω (Ideal)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "rank_vs_omega_ideal.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(omega_values, ents_ideal, marker="o")
    plt.xlabel("ω (encoding frequency)")
    plt.ylabel("Spectral Entropy")
    plt.title("Kernel Spectral Entropy vs ω (Ideal)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "entropy_vs_omega_ideal.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(omega_values, ents_noisy, marker="o")
    plt.xlabel("ω (encoding frequency)")
    plt.ylabel("Spectral Entropy")
    plt.title("Kernel Spectral Entropy vs ω (Noisy, p=0.01)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "entropy_vs_omega_noisy.png", dpi=300)
    plt.close()


def main():
    generate_dataset_scatter()
    generate_expressivity_sweep_figures()
    print("[OK] All figures generated.")


if __name__ == "__main__":
    main()
