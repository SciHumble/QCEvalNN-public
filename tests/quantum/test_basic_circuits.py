import pytest
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
from qc_eval.quantum.quantum_circuit import QuantumCircuit
from qc_eval.quantum.basic_circuits import BasicCircuits


def test_get_circuit_dict():
    circuit_dict = BasicCircuits.get_circuit_dict()
    assert isinstance(circuit_dict, dict), "Circuit dictionary is not a dict."
    # Check that both 'conv' and 'pool' keys are present.
    for key in ["conv", "pool"]:
        assert key in circuit_dict, f"Key '{key}' not found in circuit dict."
    conv_list = list(range(1, 6)) + [None]
    for conv in conv_list:
        assert conv in circuit_dict["conv"], f"Conv key {conv} not found."
    pool_list = list(range(1, 4)) + [None]
    for pool in pool_list:
        assert pool in circuit_dict["pool"], f"Pool key {pool} not found."


def test_unitary_matrix():
    params = [np.pi / 2, np.pi / 3, np.pi / 4]
    expected_matrix = np.array([
        [np.cos(params[0] / 2),
         -np.exp(1j * params[2]) * np.sin(params[0] / 2)],
        [np.exp(1j * params[1]) * np.sin(params[0] / 2),
         np.exp(1j * (params[1] + params[2])) * np.cos(params[0] / 2)]
    ], dtype=complex)
    matrix = BasicCircuits.unitary_matrix(params)
    # Check the shape and numerical equality.
    assert matrix.shape == (2, 2), "Unitary matrix shape mismatch."
    np.testing.assert_allclose(matrix, expected_matrix, atol=1e-5)


def test_unitary_matrix_incorrect_length():
    with pytest.raises(IndexError):
        _ = BasicCircuits.unitary_matrix([np.pi, 0])  # Too few parameters


@pytest.mark.parametrize("circuit_class, num_params, num_qubits", [
    (BasicCircuits.CU, 3, 2),
    (BasicCircuits.Pool1, 2, 2),
    (BasicCircuits.Pool2, 6, 2),
    (BasicCircuits.Pool3, 6, 2),
    (BasicCircuits.PoolDefault, 3, 2),
    (BasicCircuits.Conv1, 2, 2),
    (BasicCircuits.Conv2, 4, 2),
    (BasicCircuits.Conv3, 10, 2),
    (BasicCircuits.Conv4, 10, 2),
    (BasicCircuits.Conv5, 15, 2),
    (BasicCircuits.ConvDefault, 3, 2),
])
def test_circuit_classes(circuit_class, num_params, num_qubits):
    qubits = [0, 1]
    params = np.random.random(num_params)
    # Prepare a random state vector of length 2^num_qubits.
    x = np.random.random(2 ** num_qubits)
    circuit = circuit_class(qubits=qubits, parameters=params)

    assert isinstance(circuit,
                      QuantumCircuit), f"{circuit_class.__name__} is not an instance of QuantumCircuit."
    assert circuit.num_parameters == num_params, (
        f"{circuit_class.__name__}: Expected {num_params} parameters but got {circuit.num_parameters}."
    )
    assert circuit.qubits == qubits, f"{circuit_class.__name__}: Qubit configuration mismatch."

    # Create a QNode to execute the circuit.
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def qnode():
        AmplitudeEmbedding(x, wires=range(num_qubits), normalize=True)
        circuit.circuit()
        return qml.probs(wires=qubits)

    probs = qnode()
    # Check that the probabilities are nonnegative and sum to 1.
    assert np.all(
        probs >= 0), f"{circuit_class.__name__}: Negative probabilities encountered."
    np.testing.assert_allclose(np.sum(probs), 1, atol=1e-5,
                               err_msg=f"{circuit_class.__name__}: Probabilities do not sum to 1.")


@pytest.mark.parametrize("circuit_class, expected_name", [
    (BasicCircuits.CU, "controlled unitary gate"),
    (BasicCircuits.Pool1, "pooling circuit 1"),
    (BasicCircuits.Pool2, "pooling circuit 2"),
    (BasicCircuits.Pool3, "pooling circuit 3"),
    (BasicCircuits.PoolDefault, "default pooling circuit"),
    (BasicCircuits.Conv1, "convolutional circuit 1"),
    (BasicCircuits.Conv2, "convolutional circuit 2"),
    (BasicCircuits.Conv3, "convolutional circuit 3"),
    (BasicCircuits.Conv4, "convolutional circuit 4"),
    (BasicCircuits.Conv5, "convolutional circuit 5"),
    (BasicCircuits.ConvDefault, "default convolutional circuit"),
])
def test_circuit_names(circuit_class, expected_name):
    circuit = circuit_class(qubits=[0, 1])
    assert circuit.name == expected_name, f"Expected name '{expected_name}' but got '{circuit.name}'."


def test_plot_returns_figure():
    # Test that the plot() method of a representative circuit returns a
    # matplotlib Figure.
    circuit = BasicCircuits.Conv1(
        qubits=[0, 1],
        parameters=np.random.random(BasicCircuits.Conv1._num_parameters)
    )
    fig = circuit.plot()
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure), \
        "plot() did not return a matplotlib Figure instance."
