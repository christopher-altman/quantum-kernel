import torch
import pennylane as qml

"""Torch-based differentiable QML utilities."""


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_torch_embedding_model(n_qubits: int = 4, omega: float = 8.0):
    """
    Torch-compatible VQC model using default.qubit with interface='torch'.
    """
    dev = qml.device("default.qubit", wires=n_qubits, interface="torch")

    @qml.qnode(dev, interface="torch")
    def model(x, weights):
        wires = list(range(n_qubits))
        x0, x1 = x[0], x[1]

        # Simple high-frequency embedding
        for w in wires:
            qml.RZ(omega * x0, wires=w)
            qml.RX(omega * x1, wires=w)

        qml.StronglyEntanglingLayers(weights, wires=wires)
        return qml.expval(qml.PauliZ(0))

    return model


def torch_vqc_step(model, weights, X_batch, y_batch, optimizer, device=None):
    """
    Single optimization step for a Torch VQC model.
    """
    if device is None:
        device = get_torch_device()

    weights = weights.to(device)
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()
    preds = []
    for x in X_batch:
        z = model(x, weights)
        p = (1.0 + z) / 2.0
        preds.append(p)
    preds = torch.stack(preds)
    loss = torch.mean((preds - y_batch) ** 2)
    loss.backward()
    optimizer.step()
    return weights, float(loss.item())