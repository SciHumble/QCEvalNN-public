import pytest
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from qc_eval.quantum.qcnn import QCNN
from qc_eval.quantum.quantum_circuit import QuantumCircuit
from qc_eval.quantum.basic_circuits import BasicCircuits


# --- Initialization tests ---

def test_qcnn_init_no_noise():
    num_qubits = 2
    num_layers = 1
    conv_circ = 1  # Should select BasicCircuits.Conv1
    pool_circ = 1  # Should select BasicCircuits.Pool1
    # According to _calc_num_parameters,
    # total parameters = num_layers * (conv_circ.num_parameters * 2)
    # and in the current implementation,
    # conv_circ.num_parameters returns 2; hence, total=2+2=4.
    params = np.random.random(4).tolist()
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    # Check that qubits are correctly set.
    assert qcnn.qubits == list(range(num_qubits))
    # Check device type via class name.
    assert qcnn.device.__class__.__name__ == "DefaultQubit", ("Expected "
                                                              "device to be "
                                                              "DefaultQubit "
                                                              "when no noise "
                                                              "is provided.")
    assert qcnn.parameters == params


def test_qcnn_init_with_noise():
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    noise = {"single gate": 0.05, "cnot": 0.1}
    params = np.random.random(4).tolist()
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=noise,
                params=params)
    # When noise is provided, the device should be of type DefaultMixed.
    assert qcnn.device.__class__.__name__ == "DefaultMixed", ("Expected "
                                                              "device to be "
                                                              "DefaultMixed "
                                                              "when noise is "
                                                              "provided.")
    assert qcnn.noise == noise


# --- Parameter Calculation and Splitting tests ---

def test_calc_num_parameters():
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    params = np.random.random(4).tolist()
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    # In the current implementation,
    # total parameters = 2*(conv_circ.num_parameters).
    # According to observed behavior, conv_circ.num_parameters is 2,
    # so total = 4.
    assert qcnn._calc_num_parameters() == 4


@pytest.mark.parametrize(
    "num_layers, conv_len, pool_len",
    [
        (1, 2, 2),
        (2, 3, 1),
        (3, 1, 1),
        (2, 2, 3),
        (1, 0, 2),  # Edge case: no conv params
        (1, 2, 0),  # Edge case: no pool params
    ]
)
def test_prepare_params_multi_layer(num_layers, conv_len, pool_len, monkeypatch):
    """
    Test _prepare_params for different layer counts and conv/pool parameter lengths.
    """
    print(num_layers, conv_len, pool_len)
    total_params_per_layer = conv_len + pool_len
    total_params = num_layers * total_params_per_layer

    # Create a flat parameter list: [0.0, 0.1, 0.2, ..., n]
    params = [round(i * 0.1, 1) for i in range(total_params)]

    # Create dummy QCNN-like object with mocked circuit param sizes
    monkeypatch.setattr(BasicCircuits.ConvDefault, "_num_parameters", conv_len)
    monkeypatch.setattr(BasicCircuits.PoolDefault, "_num_parameters", pool_len)
    qcnn = QCNN(
        num_qubits=8,  # must be at least 8 for 3 layers
        num_layers=num_layers,
        conv_circ=None,
        pool_circ=None,
        noise=None,
        params=params
    )

    conv_params, pool_params = qcnn._prepare_params()

    # Expected conv and pool param groups
    expected_conv = []
    expected_pool = []

    index = 0
    for _ in range(num_layers):
        expected_conv.append(params[index:index + conv_len])
        index += conv_len
        expected_pool.append(params[index:index + pool_len])
        index += pool_len

    assert conv_params == expected_conv, (
        f"Conv params mismatch:\nExpected: {expected_conv}\nGot: {conv_params}"
    )
    assert pool_params == expected_pool, (
        f"Pool params mismatch:\nExpected: {expected_pool}\nGot: {pool_params}"
    )


# --- Data Embedding tests ---

def test_data_embedding_amplitude():
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    params = np.random.random(4).tolist()
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    # For amplitude embedding, the input must be of length 2^num_qubits = 4.
    x = np.array([0.1, 0.2, 0.3, 0.4])
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def qnode():
        qcnn._data_embedding(x, embedding_type="amplitude")
        return qml.probs(wires=range(num_qubits))

    probs = qnode()
    np.testing.assert_allclose(np.sum(probs), 1, atol=1e-5)


def test_data_embedding_angle():
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    params = np.random.random(4).tolist()
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    # For angle embedding, the input must be of length <= num_qubits;
    # we supply a vector of length 2.
    x = np.array([0.1, 0.2])
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def qnode():
        qcnn._data_embedding(x, embedding_type="angle")
        return qml.probs(wires=range(num_qubits))

    probs = qnode()
    np.testing.assert_allclose(np.sum(probs), 1, atol=1e-5)


def test_data_embedding_invalid():
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    params = np.random.random(4).tolist()
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    dev = qml.device("default.qubit", wires=num_qubits)
    with pytest.raises(NotImplementedError):
        @qml.qnode(dev)
        def qnode():
            qcnn._data_embedding(x, embedding_type="invalid")
            return qml.probs(wires=range(num_qubits))

        qnode()


# --- qnode method tests ---

def test_qnode_mse():
    # Test that the qnode method with cost_function "mse" returns a scalar.
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    params = [0.1, 0.2, 0.3, 0.4]  # total expected length is 4.
    x = np.array([0.1, 0.2, 0.3, 0.4])
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    result = qcnn.qnode(x, params, embedding="amplitude", cost_function="mse")
    # "mse" branch returns an expectation value (scalar or 0D array).
    assert np.isscalar(result) or (
            isinstance(result, np.ndarray) and result.ndim == 0)


def test_qnode_cross_entropy():
    # Test that the qnode method with cost_function "cross entropy"
    # returns a probability vector.
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    params = [0.1, 0.2, 0.3, 0.4]
    x = np.array([0.1, 0.2, 0.3, 0.4])
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    result = qcnn.qnode(x, params, embedding="amplitude",
                        cost_function="cross entropy")
    # The "cross entropy" branch returns probabilities from a single-qubit
    # measurement (wire 0),
    # which should be a 1D array of length 2.
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.size == 2


def test_qnode_invalid_cost():
    # Test that an invalid cost function raises NotImplementedError.
    num_qubits = 2
    num_layers = 1
    conv_circ = 1
    pool_circ = 1
    params = [0.1, 0.2, 0.3, 0.4]
    x = np.array([0.1, 0.2, 0.3, 0.4])
    qcnn = QCNN(num_qubits=num_qubits, num_layers=num_layers,
                conv_circ=conv_circ, pool_circ=pool_circ, noise=None,
                params=params)
    with pytest.raises(NotImplementedError):
        qcnn.qnode(x, params, embedding="amplitude", cost_function="invalid")


def test_property_num_parameters():
    qcnn = QCNN(num_qubits=4, num_layers=None, conv_circ=1, pool_circ=1)
    num_params = qcnn.num_parameters
    assert num_params > 0
    assert isinstance(num_params, int)
