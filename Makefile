PYTHON := python

.PHONY: help install dev test figures paper package clean

help:
	@echo "Targets:"
	@echo "  install   - pip install in editable mode"
	@echo "  dev       - install dev extras"
	@echo "  test      - quick smoke test (dataset + one kernel)"
	@echo "  figures   - generate main figures into diagrams/"
	@echo "  paper     - export notebooks to HTML/PDF and build LaTeX"
	@echo "  package   - create repo ZIP archive in dist/"
	@echo "  clean     - remove build artifacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -c "from src.dataset_c4 import generate_hybrid_dataset; X,y,_=generate_hybrid_dataset(); print('Dataset OK', X.shape, 'labels', set(y))"
	$(PYTHON) -c "from src.quantum_kernels import make_kernel_circuit, compute_kernel_matrix; import numpy as np; import sklearn.model_selection as ms; from sklearn.svm import SVC; from src.dataset_c4 import generate_hybrid_dataset; X,y,_=generate_hybrid_dataset(); Xtr,Xte,ytr,yte=ms.train_test_split(X,y,test_size=0.3,random_state=42,stratify=y); k=make_kernel_circuit(4,omega=5.0); Ktr=compute_kernel_matrix(k,Xtr,Xtr); Kte=compute_kernel_matrix(k,Xte,Xtr); clf=SVC(kernel='precomputed'); clf.fit(Ktr,ytr); import sklearn.metrics as M; print('QKS acc', M.accuracy_score(yte, clf.predict(Kte)))"

figures:
	$(PYTHON) scripts/generate_figures.py

paper:
	$(PYTHON) scripts/export_notebooks.py
	cd docs 2>/dev/null || mkdir docs
	pdflatex -interaction=nonstopmode neurips_extended_abstract.tex || echo "LaTeX build requires TeX installed."

package:
	$(PYTHON) scripts/package_repo.py

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
