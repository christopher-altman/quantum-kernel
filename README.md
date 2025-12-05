# Quantum Kernel Expressivity: Measuring Inductive Bias and Feature Complexity in Quantum Kernel Methods

This repository investigates how expressivity in quantum kernels correlates with feature–space complexity, generalization behavior, and spectral properties of the associated kernel matrices across varying circuit depths, entangling structures, and feature encodings.

The implementation is built using PennyLane, PyTorch, and custom kernel-simulation utilities.

This repository contains reproducible implementations of:

- quantum kernel estimators for multiple feature maps
- expressivity metrics including rank, spectrum, and capacity measures
- controlled experiments comparing depth, width, and data regimesdata regimes

All experiments can be executed directly via the provided scripts and notebooks. Default configurations reproduce the key expressivity results without modification.

___

## Abstract

Quantum kernel methods provide a mechanism for embedding classical inputs into a high-dimensional feature space induced by parameterized quantum circuits. The expressivity of this embedding—determined by the structure, depth, and entanglement properties of the circuit—directly influences generalization performance, inductive bias, and sample complexity.

This repository explores these effects by constructing and analyzing multiple kernel ansätze, computing their induced Gram matrices, and evaluating how expressivity metrics vary with architectural parameters. We quantify changes in spectral signatures, rank profiles, embedding smoothness, and the geometry of feature space as circuit complexity increases.

___

## Methods and Contributions

This repository implements a reproducible expressivity benchmarking framework consisting of:

### Circuit and Kernel Variants
  - shallow (low-depth) vs deep feature-map circuits
  - deep feature maps with full entanglement layers
	- different entangling layouts (linear, full, block entanglement)
	- deterministic vs stochastic feature encoding mechanisms

### Expressivity Metrics Evaluated
  - eigenvalue distribution and spectral decay  
	- effective rank and trace-norm computation of Gram matrices
	- empirical concentration effects under repeated sampling
  - margin-based feature separation metrics
	- curvature in feature-space embeddings

### Core Experimental Contributions
  - side-by-side expressivity benchmarks across circuit configuration families
  - reproducible procedure for computing spectral expressivity metrics  
  - interpretable plots and spectral visualization pipeline
  - a complete execution pipeline suitable for research replication
  - modular design enabling new kernel families  

Implementations are built on PennyLane simulators with PyTorch integration for efficient batch processing execution.

## Repository Structure

```
├── src/
│   ├── kernels/                # Kernel constructors
│   ├── circuits/               # Feature map architectures
│   ├── metrics/                # Rank, spectrum, trace-norm utilities
│   ├── datasets/               # Synthetic dataset loaders
│   └── experiments/            # Orchestrated experiment routines
│
├── notebooks/
│   ├── exploratory/            # Interactive exploration notebooks
│   └── figures.ipynb           # Notebook to generate final figures
│
├── figures/                    # Generated plots, heatmaps, distributions
├── diagrams/                   # UML circuit diagrams or architecture sketches
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

This structure is intentionally modular to support expansion into new circuit families and new kernel metrics.

---

## Experiment Suite

| Experiment ID | Circuit Family            | Parameter Varied        | Metric Evaluated                        | Artifact Output               |
|---------------|---------------------------|--------------------------|------------------------------------------|-------------------------------|
| EXP-01        | Shallow feature map       | depth {1..4}             | eigenvalue spectrum decay                | eigenvalue curves             |
| EXP-02        | Deep entangling structure | entanglement radius      | trace norm vs effective rank             | heatmaps, diagnostics         |
| EXP-03        | Random phase encoding     | randomness strength      | kernel smoothness & separability         | margin separation curves      |
| EXP-04        | Structured encoding       | dimensionality scaling   | generalization gap proxy                 | performance trend curves      |

Outputs are stored in `figures/`, or optionally `results/` when using batch execution.

---

## Installation & Usage

### Create environment
```
python3 -m venv .venv
source .venv/bin/activate
```
### Install dependencies
```
pip install -r requirements.txt
```
### Run example experiment
```
python src/experiments/run_expressivity.py –depth 4 –entanglement full
```
### Generate figures
```
python src/experiments/generate_figures.py
```
### Launch notebooks
```
jupyter lab
```
---

## Citation
```
@misc{quantum_kernel_expressivity,
  title    = {Quantum Kernel Expressivity: Measuring Inductive Bias and Feature Complexity in Quantum Kernel Methods},
  author   = {Altman, Christopher},
  year     = {2025},
  note     = {GitHub repository},
  url      = {https://github.com/quantum-kernel},
  urldate  = {2025-12-06},
}
```


