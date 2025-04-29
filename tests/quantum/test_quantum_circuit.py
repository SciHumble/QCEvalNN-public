import pennylane as qml
from qc_eval.quantum.quantum_circuit import QuantumCircuit


# Create a concrete dummy subclass of QuantumCircuit for testing.
class DummyQuantumCircuit(QuantumCircuit):
    def __init__(self, qubits: list[int], parameters: list[float] = None,
                 noise: dict = None):
        self.name = "DummyQuantumCircuit"  # required by QuantumCore
        super().__init__(qubits, parameters, noise)

    def circuit(self) -> float:
        # If noise is provided (error_rate is a dict), add noise.
        if self.error_rate is not None:
            self._add_single_gate_noise(self.qubits[0])
            self._add_cnot_noise(self.qubits)
        # Apply a simple RX gate so that the circuit returns an expectation value.
        qml.RX(0.1, wires=self.qubits[0])
        return qml.expval(qml.PauliZ(self.qubits[0]))

    def _circuit_logger_message(self):
        # Inherit the default behavior from QuantumCircuit.
        super()._circuit_logger_message()


# -------------------------
# Tests for __init__ behavior.
# -------------------------
def test_default_parameters():
    # When no parameters are provided, they should default to zeros of length num_parameters (3).
    qc = DummyQuantumCircuit(qubits=[0, 1])
    assert qc.parameters == [0, 0, 0]


def test_noise_initialization():
    noise = {"single gate": 0.05, "cnot": 0.1}
    qc = DummyQuantumCircuit(qubits=[0, 1], noise=noise)
    assert isinstance(qc.error_rate, dict)
    assert qc.error_rate["single gate"] == 0.05
    assert qc.error_rate["cnot"] == 0.1


def test_no_noise_initialization():
    qc = DummyQuantumCircuit(qubits=[0, 1], noise=None)
    assert qc.error_rate is None


# -------------------------
# Tests for noise functions via the tape.
# -------------------------
def test_circuit_with_noise():
    noise = {"single gate": 0.05, "cnot": 0.1}
    qc = DummyQuantumCircuit(qubits=[0, 1], noise=noise)
    # Use a device that supports noise channels.
    dev = qml.device("default.mixed", wires=qc.num_qubits)

    @qml.qnode(dev)
    def circuit_qnode():
        return qc.circuit()

    _ = circuit_qnode()  # Execute the circuit.
    tape = circuit_qnode.tape
    # Find all operations with name "DepolarizingChannel".
    dep_ops = [op for op in tape.operations if
               op.name == "DepolarizingChannel"]
    # _add_single_gate_noise is called once and _add_cnot_noise is called
    # for each qubit in qc.qubits.
    # With qc.qubits == [0, 1], we expect 1 + 2 = 3 DepolarizingChannel
    # operations.
    assert len(dep_ops) == 3
    # Check that one operation has probability 0.05 and two have 0.1.
    probs = [op.parameters[0] for op in dep_ops]
    assert 0.05 in probs
    assert probs.count(0.1) == 2


def test_circuit_without_noise():
    qc = DummyQuantumCircuit(qubits=[0, 1], noise=None)
    dev = qml.device("default.mixed", wires=qc.num_qubits)

    @qml.qnode(dev)
    def circuit_qnode():
        return qc.circuit()

    _ = circuit_qnode()
    tape = circuit_qnode.tape
    dep_ops = [op for op in tape.operations if
               op.name == "DepolarizingChannel"]
    assert len(dep_ops) == 0


# -------------------------
# Test for _circuit_logger_message.
# -------------------------
def test_circuit_logger_message():
    qc = DummyQuantumCircuit(qubits=[0, 1])
    # Simply call the method to verify that it executes without error.
    qc._circuit_logger_message()
