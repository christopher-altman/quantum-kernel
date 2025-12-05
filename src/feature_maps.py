import pennylane as qml

"""High-frequency and temporal feature maps for QML experiments."""


def high_frequency_feature_map(x, wires, omega: float = 8.0):
    x0, x1 = x[0], x[1]
    for w in wires:
        qml.RZ(omega * x0, wires=w)
        qml.RX(omega * x1, wires=w)
        qml.RY(0.5 * omega * x0 * x1, wires=w)
    num_wires = len(wires)
    for i in range(num_wires):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % num_wires]])


def temporal_feature_map(x, t, wires, omega_t: float = 2.0):
    high_frequency_feature_map(x, wires, omega=omega_t)
    for w in wires:
        qml.RZ(t, wires=w)
