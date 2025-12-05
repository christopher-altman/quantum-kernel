from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

"""Classical ML baselines."""


def classical_rbf_svm(C: float = 1.0, gamma="scale", probability: bool = False):
    return SVC(kernel="rbf", C=C, gamma=gamma, probability=probability)


def classical_poly_svm(C: float = 1.0, degree: int = 3, coef0: float = 1.0):
    return SVC(kernel="poly", C=C, degree=degree, coef0=coef0)


def classical_logistic(max_iter: int = 1000):
    return LogisticRegression(max_iter=max_iter)
