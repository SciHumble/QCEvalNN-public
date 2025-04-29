from qc_eval.quantum.quantum_core import QuantumCore
import pytest
import matplotlib.pyplot as plt
import pennylane as qml


# Dummy concrete implementation for testing.
class DummyQuantumCore(QuantumCore):
    def __init__(self, name="dummy", num_qubits=2, num_parameters=1):
        self.name = name
        self.num_qubits = num_qubits
        self._num_parameters = num_parameters
        # Initialize qubits as [0, 1, ...]
        self.qubits = list(range(num_qubits))
        # Initialize parameters with default values.
        self.parameters = [0.5] * num_parameters
        self._gates = None

    def circuit(self) -> None:
        # Apply an RX gate using the first parameter.
        qml.RX(self.parameters[0], wires=self.qubits[0])
        # If more than one qubit, add a CNOT gate.
        if self.num_qubits > 1:
            qml.CNOT(wires=[self.qubits[0], self.qubits[1]])


# -------------------------
# Tests for the _check_length method.
# -------------------------
def test_check_length_too_short():
    with pytest.raises(ValueError):
        QuantumCore._check_length([1, 2], expected_length=3,
                                  property_name="test_property")


def test_check_length_too_long():
    result = QuantumCore._check_length([1, 2, 3, 4], expected_length=3,
                                       property_name="test_property")
    assert len(result) == 3


# -------------------------
# Tests for the parameters setter.
# -------------------------
def test_parameters_setter_correct():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=2)
    qc.parameters = [0.1, 0.2]
    assert qc.parameters == [0.1, 0.2]


def test_parameters_setter_too_short():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=3)
    with pytest.raises(ValueError):
        qc.parameters = [0.1, 0.2]


def test_parameters_setter_too_long():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=2)
    qc.parameters = [0.1, 0.2, 0.3, 0.4]
    assert qc.parameters == [0.1, 0.2]


# -------------------------
# Tests for the qubits setter.
# -------------------------
def test_qubits_setter_correct():
    qc = DummyQuantumCore(num_qubits=3, num_parameters=1)
    qc.qubits = [0, 1, 2]
    assert qc.qubits == [0, 1, 2]


def test_qubits_setter_too_short():
    qc = DummyQuantumCore(num_qubits=3, num_parameters=1)
    with pytest.raises(ValueError):
        qc.qubits = [0, 1]


def test_qubits_setter_too_long():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=1)
    qc.qubits = [0, 1, 2, 3]
    assert qc.qubits == [0, 1]


# -------------------------
# Test for the plot method.
# -------------------------
def test_plot_returns_figure():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=1)
    fig = qc.plot()
    assert isinstance(fig, plt.Figure)


# -------------------------
# Tests for the gates and num_gates properties.
# -------------------------
def test_gates_property():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=1)
    gates = qc.gates
    # Expect one RX and one CNOT.
    assert gates.get("RX", 0) == 1
    assert gates.get("CNOT", 0) == 1
    assert set(gates.keys()) == {"RX", "CNOT"}


def test_num_gates_property():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=1)
    # We expect total 2 gates.
    assert qc.num_gates == 2


# -------------------------
# Test for the num_parameters property.
# -------------------------
def test_num_parameters_property():
    qc = DummyQuantumCore(num_qubits=2, num_parameters=3)
    assert qc.num_parameters == 3
