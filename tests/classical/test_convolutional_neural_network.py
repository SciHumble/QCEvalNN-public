import pytest
import torch
import torch.nn as nn
from collections import OrderedDict
from qc_eval.classical.convolutional_neural_network import ClassicalConvNN
from unittest.mock import patch


def test_initialization():
    sequence_length = 64
    num_layers = 3
    cnn = ClassicalConvNN(sequence_length=sequence_length,
                          num_layers=num_layers)

    assert cnn.sequence_length == sequence_length, \
        "Sequence length not set correctly."
    assert cnn.num_layers == num_layers, \
        "Number of layers not set correctly."
    assert len(cnn.convs) == num_layers, \
        "Incorrect number of convolution layers."
    assert len(cnn.pools) == num_layers, \
        "Incorrect number of pooling layers."


def test_forward_pass():
    sequence_length = 64
    num_layers = 1
    cnn = ClassicalConvNN(sequence_length=sequence_length,
                          num_layers=num_layers)
    dummy_input = torch.randn(1, 1, sequence_length)
    # Note: The output shape from ClassicalConvNN.forward() is
    # (batch_size, 1, expected_length)
    assert output_shape(dummy_input, cnn) == (1, 2), \
        "Output shape is incorrect."


def output_shape(x, cnn):
    with torch.no_grad():
        out = cnn.forward(x)
    return tuple(out.shape)


def test_initialization_logging():
    sequence_length = 64
    num_layers = 3
    with patch(
            'qc_eval.classical.convolutional_neural_network.logger'
    ) as mock_logger:
        _ = ClassicalConvNN(sequence_length=sequence_length,
                            num_layers=num_layers)
        assert mock_logger.debug.called, \
            "Logging not called during initialization."
        # Expect at least one debug call during initialization.
        assert mock_logger.debug.call_count >= num_layers + 1, \
            "Incorrect number of log calls."


# === New tests for properties ===

def test_num_parameters_property():
    sequence_length = 64
    num_layers = 3
    cnn = ClassicalConvNN(sequence_length=sequence_length,
                          num_layers=num_layers)
    total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    assert cnn.num_parameters == total_params, \
        "num_parameters property does not match the computed total."


def test_ordered_dict_property():
    sequence_length = 64
    num_layers = 3
    cnn = ClassicalConvNN(sequence_length=sequence_length,
                          num_layers=num_layers)
    od = cnn.ordered_dict
    assert isinstance(od, OrderedDict), \
        "ordered_dict property is not an OrderedDict."
    # For each layer we expect keys:
    # conv{i}, relu{i}, pool{i}; plus "flatten" and "linear"
    expected_keys = []
    for i in range(num_layers):
        expected_keys.extend([f"conv{i}", f"relu{i}", f"pool{i}"])
    expected_keys.extend(["flatten", "linear"])
    assert list(od.keys()) == expected_keys, \
        f"ordered_dict keys mismatch. Got: {list(od.keys())}"


def test_ccnn_property():
    sequence_length = 64
    num_layers = 3
    cnn = ClassicalConvNN(sequence_length=sequence_length,
                          num_layers=num_layers)
    seq = cnn.ccnn
    assert isinstance(seq, nn.Sequential), \
        "ccnn property does not return an nn.Sequential."
    # The expected number of modules is num_layers*3
    # (conv, relu, pool for each layer) + 2 (flatten and linear).
    expected_num_modules = num_layers * 3 + 2
    actual_num_modules = len(list(seq.children()))
    assert actual_num_modules == expected_num_modules, \
        (f"Expected {expected_num_modules} modules in ccnn, "
         f"got {actual_num_modules}.")


# Running tests if executed as script.
if __name__ == "__main__":
    pytest.main()
