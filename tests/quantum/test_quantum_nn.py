import numpy as np
import pennylane as qml
from qc_eval.quantum.quantum_nn import QuantumNN


# Define a dummy concrete subclass of QuantumNN.
class DummyQuantumNN(QuantumNN):
    def __init__(self, qubits: list[int], parameters: list[float] = None,
                 noise: dict = None, num_layers: int = 1, device=None):
        # Set required attributes.
        self.name = "DummyQuantumNN"
        self.num_layers = num_layers
        self.noise = noise
        self.device = device
        # Set number of qubits from the provided list.
        self.num_qubits = len(qubits)
        # If no parameters are provided, default to a single parameter 0.
        if parameters is None:
            self._num_parameters = 1
            parameters = [0.0]
        else:
            self._num_parameters = len(parameters)
        self.qubits = qubits
        self.parameters = parameters

    def circuit(self) -> float:
        # A dummy circuit implementation for QuantumCore compatibility.
        # Here we simply apply an RX gate with a zero angle and return the expectation of PauliZ.
        qml.RX(0.0, wires=self.qubits[0])
        return qml.expval(qml.PauliZ(self.qubits[0]))

    def qnode(self, x, params, embedding, cost_function):
        # Use the provided device if available; otherwise, default to "default.qubit".
        dev = self.device if self.device is not None else qml.device(
            "default.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit_qnode(x):
            # Compute an angle as the sum of:
            #   - the first parameter,
            #   - the result of cost_function(x),
            #   - the result of embedding(x)
            angle = params[0] + cost_function(x) + embedding(x)
            qml.RX(angle, wires=self.qubits[0])
            return qml.expval(qml.PauliZ(self.qubits[0]))

        return circuit_qnode(x)


# Define simple dummy functions for embedding and cost.
def dummy_embedding(x):
    # For testing, simply return x.
    return x


def dummy_cost(x):
    # For testing, return a constant value.
    return 0.1


# -------------------------
# Tests for the qnode method.
# -------------------------
def test_qnode_returns_value():
    # Create an instance with one qubit and one parameter.
    nn = DummyQuantumNN(qubits=[0], parameters=[0.2])
    x = 0.5
    result = nn.qnode(x, nn.parameters, dummy_embedding, dummy_cost)
    # The expected angle is: 0.2 + dummy_cost(x) + dummy_embedding(x)
    expected_angle = 0.2 + dummy_cost(x) + dummy_embedding(
        x)  # 0.2 + 0.1 + 0.5 = 0.8
    # For an RX rotation, the expectation value of PauliZ is cos(angle).
    expected = np.cos(expected_angle)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_qnode_with_custom_device():
    # Create a device explicitly.
    dev = qml.device("default.qubit", wires=1)
    nn = DummyQuantumNN(qubits=[0], parameters=[0.3], device=dev)
    x = 0.2
    result = nn.qnode(x, nn.parameters, dummy_embedding, dummy_cost)
    expected_angle = 0.3 + dummy_cost(x) + dummy_embedding(x)
    expected = np.cos(expected_angle)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
