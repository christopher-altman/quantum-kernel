"""Generate noise sensitivity curve for a fixed ω."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.dataset_c4 import generate_hybrid_dataset
from src.quantum_kernels import make_noisy_kernel_circuit, compute_kernel_matrix

ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = ROOT / "diagrams"
DIAGRAMS_DIR.mkdir(exist_ok=True, parents=True)

X, y, meta = generate_hybrid_dataset(n_samples=500, seed=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

omega_star = 10.0
noise_vals = np.linspace(0.0, 0.08, 9)
acc = []

for nl in noise_vals:
    kernel_fn = make_noisy_kernel_circuit(4, omega_star, noise_level=nl)
    K_train = compute_kernel_matrix(kernel_fn, X_train, X_train)
    K_test = compute_kernel_matrix(kernel_fn, X_test, X_train)

    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    acc.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(7, 5))
plt.plot(noise_vals, acc, marker="o")
plt.xlabel("Noise probability p")
plt.ylabel("Accuracy")
plt.title(f"Sensitivity to Depolarizing Noise at ω={omega_star}")
plt.grid(True)
plt.tight_layout()
out = DIAGRAMS_DIR / "accuracy_vs_noise.png"
plt.savefig(out, dpi=300)
plt.close()

print(f"[OK] Saved noise curve -> {out}")
