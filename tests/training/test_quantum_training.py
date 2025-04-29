import os
import numpy as np
import pytest
import torch
import pennylane as qml
from qc_eval.training.quantum_training import QTraining, generate_noise_dict
from qc_eval.dataset.dataset import Dataset
from qc_eval.training.training import Training
from typing import Optional
import datetime


# Dummy dataset for quantum training tests.
class DummyQuantumDataset(Dataset):
    def __init__(self, feature_reduction="dummy", classes=None,
                 quantum_output=False, compact=False, **kwargs):
        self.feature_reduction = feature_reduction
        self.name = "DummyQuantumDataset"
        # For simplicity, use 4 training samples and 2 test samples.
        self.x_train = np.array([0, 1, 2, 3])
        self.y_train = [1, 0, 1, 0]
        self.x_test = np.array([4, 5])
        self.y_test = [1, 0]

    def dataset(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def convert_to_array(self):
        x_array = np.array([self.x_train, self.x_test], dtype=object)
        y_array = np.array([self.y_train, self.y_test], dtype=object)
        return np.array([x_array, y_array], dtype=object)


# Dummy QCNN class to simulate a quantum neural network.
class DummyQCNN:
    def __init__(self, **kwargs):
        self.num_parameters = 3
        self.num_qubits = kwargs.get("number_of_qubits", 4)

    def qnode(self, x, params, embedding, cost_function):
        # Always return a fixed dummy probability distribution.
        return [0.3, 0.7]

    @property
    def circuit(self):
        # Return a dummy circuit (not used in these tests).
        return lambda: None

    @property
    def gates(self):
        return {"DummyGate": 1}


# Dummy QTraining subclass that overrides __init__ to use DummyQCNN.
class DummyQTraining(QTraining):
    def __init__(self, number_of_qubits, dataset, **kwargs):
        self.number_of_qubits = number_of_qubits
        self.dataset = dataset
        self.embedding_type = kwargs.get("embedding_type", "amplitude")
        self.cost_function = kwargs.get("cost_function", "mse")
        self._qcnn_init_parameters = {
            "number_of_qubits": number_of_qubits,
            "number_of_layers": 1,
            "convolutional_circuit_number": kwargs.get(
                "convolutional_circuit_number", 1),
            "pooling_circuit_number": kwargs.get("pooling_circuit_number", 1),
            "noise": kwargs.get("noise", None)
        }
        # Use DummyQCNN instead of a real QCNN.
        self.quantum_circuit = DummyQCNN(**self._qcnn_init_parameters)
        self._neural_network = None
        self._optimizer_name = "nesterov"
        self._opt = qml.NesterovMomentumOptimizer(stepsize=0.1)
        self._parameters = np.random.randn(self.quantum_circuit.num_parameters)
        self._loss_history = []
        self.timestamp = "dummy_timestamp"
        # For testing, set a small number of training steps and batch size.
        self._steps = 2
        self._batch_size = 2
        self._save_rate = 1

    def train(self, finished_epoch: int = 0) -> list:
        for _ in range(finished_epoch, self._steps):
            self._loss_history.append(0.5)
        return self._loss_history

    def _create_checkpoint(self, epoch: int = None) -> dict:
        return {
            "number_of_qubits": self.quantum_circuit.num_qubits,
            "dataset": self.dataset.convert_to_array(),
            "dataset_name": self.dataset.name,
            "model_init_parameters": self._qcnn_init_parameters,
            "parameters": self._parameters,
            "optimizer_momentum": 0.9,
            "optimizer": self._optimizer_name,
            "embedding_type": self.embedding_type,
            "cost_function": self.cost_function,
            "loss_history": self._loss_history,
            "epoch": epoch
        }

    def load(self, file_location: str) -> None:
        checkpoint = torch.load(file_location)
        self._parameters = checkpoint["parameters"]
        self._opt.momentum = checkpoint["optimizer_momentum"]

    def save(self, file_location: str) -> None:
        checkpoint = self._create_checkpoint(self._steps - 1)
        torch.save(checkpoint, file_location)

    @classmethod
    def init_from_save(cls, file_location: str) -> 'QTraining':
        checkpoint = torch.load(file_location, weights_only=False)
        instance = cls(number_of_qubits=checkpoint["number_of_qubits"],
                       dataset=DummyQuantumDataset())
        instance._qcnn_init_parameters = checkpoint["model_init_parameters"]
        instance._parameters = checkpoint["parameters"]
        instance._loss_history = checkpoint["loss_history"]
        instance.timestamp = "loaded_timestamp"
        return instance

    def result(self) -> dict:
        # Use cost_function "cross entropy" to avoid type issues.
        _, x_test, y_test, _ = self.dataset.dataset()
        predictions = [self.quantum_circuit.qnode(x, self._parameters,
                                                  self.embedding_type,
                                                  self.cost_function) for x in
                       x_test]
        if self.cost_function == "mse":
            loss = Training._square_loss(y_test, predictions)
        elif self.cost_function == "cross entropy":
            loss = Training._cross_entropy(y_test, predictions)
        else:
            loss = None
        accuracy = 0.75
        result = {
            "loss": loss,
            "accuracy": accuracy,
            "predictions": predictions,
            "labels": list(y_test),
            "gates": self.quantum_circuit.gates,
            "parameters": self._parameters,
            "date": self.timestamp
        }
        result.update(self.expectation_accuracy(predictions, y_test))
        return result


# --- Tests ---

# --- Dummy classes for testing ---

# Dummy dataset: minimal implementation.
class DummyQuantumDataset(Dataset):
    def __init__(self, feature_reduction="dummy", classes=None,
                 quantum_output=False, compact=False, **kwargs):
        self.feature_reduction = feature_reduction
        self.name = "DummyQuantumDataset"
        # Use 4 training samples and 2 test samples.
        self.x_train = np.array([10, 20, 30, 40])
        self.y_train = [1, 0, 1, 0]
        self.x_test = np.array([50, 60])
        self.y_test = [1, 0]

    def dataset(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def convert_to_array(self):
        x_array = np.array([self.x_train, self.x_test], dtype=object)
        y_array = np.array([self.y_train, self.y_test], dtype=object)
        return np.array([x_array, y_array], dtype=object)


# Dummy QCNN: returns a fixed probability distribution.
class DummyQCNN:
    def __init__(self, **kwargs):
        self.num_parameters = 3
        self.num_qubits = kwargs.get("number_of_qubits", 4)

    def qnode(self, x, params, embedding, cost_function):
        # Return a dummy probability distribution (two-element list)
        # so that cost functions that expect subscriptable predictions work.
        return [0.3, 0.7]

    @property
    def circuit(self):
        return lambda: None

    @property
    def gates(self):
        return {"DummyGate": 1}


# Dummy optimizer to simulate the optimizer step.
class DummyOptimizer:
    def __init__(self, stepsize=0.1):
        self.stepsize = stepsize
        self.momentum = 0.0

    def step_and_cost(self, cost_fn, params):
        # Simulate an optimizer update: new parameters are params * 0.9
        # and return a fixed dummy cost (e.g., 0.42).
        new_params = params * 0.9
        cost = 0.42
        return new_params, cost


# Concrete subclass of QTraining that uses the real train() implementation.
class ConcreteQTraining(QTraining):
    def __init__(self, number_of_qubits, dataset, **kwargs):
        self.number_of_qubits = number_of_qubits
        self.dataset = dataset
        self.embedding_type = kwargs.get("embedding_type", "amplitude")
        self.cost_function = kwargs.get("cost_function", "mse")
        self._qcnn_init_parameters = {
            "number_of_qubits": number_of_qubits,
            "number_of_layers": 1,
            "convolutional_circuit_number": kwargs.get(
                "convolutional_circuit_number", 1),
            "pooling_circuit_number": kwargs.get("pooling_circuit_number", 1),
            "noise": kwargs.get("noise", None)
        }
        # Use DummyQCNN to simulate the quantum circuit.
        self.quantum_circuit = DummyQCNN(**self._qcnn_init_parameters)
        self._neural_network = None  # Not used in training.
        self._optimizer_name = "nesterov"
        self._opt = DummyOptimizer(stepsize=0.1)
        # Set initial parameters to a known value.
        self._parameters = np.array([1.0, 2.0, 3.0])
        self._loss_history = []
        self.timestamp = "test_timestamp"
        # For testing, set a known number of training iterations and
        # batch size.
        self._steps = 3  # e.g., 3 training iterations
        self._batch_size = 2  # batch size 2
        self._save_rate = 1  # checkpoint every iteration

    def _create_checkpoint(self, epoch: Optional[int] = None) -> dict:
        return {"number_of_qubits": self.number_of_qubits,
                "model_init_parameters": self._qcnn_init_parameters,
                "parameters": self._parameters,
                "loss_history": self._loss_history,
                "epoch": epoch}

    def save(self, file_location: str) -> None:
        # For testing, simply do nothing.
        pass

    @classmethod
    def init_from_save(cls, file_location: str) -> 'QTraining':
        # For testing, return a new instance with fixed attributes.
        return cls(number_of_qubits=4, dataset=DummyQuantumDataset())


# --- Tests for the train() method ---

def test_train_tightly(monkeypatch):
    """
    Test the central train() method of QTraining.
    We monkeypatch external functions:
      - pnp.random.randint to return a fixed batch index array.
      - The optimizer's step_and_cost to return controlled updated parameters and cost.
      - _save_checkpoint is patched to do nothing (and count its calls).
    Then we verify that:
      - The loss history has the expected length.
      - The final parameters equal the initial parameters multiplied by 0.9^_steps.
    """
    # Create a controlled dummy dataset.
    dataset = DummyQuantumDataset()
    # Instantiate our concrete training class.
    trainer = ConcreteQTraining(number_of_qubits=4, dataset=dataset,
                                cost_function="cross entropy")
    initial_params = trainer._parameters.copy()

    # Patch the random batch generation to always return indices [0, 1].
    # QTraining uses: pnp.random.randint(0, len(x_train), (self._batch_size,))
    monkeypatch.setattr(qml.numpy.random, "randint",
                        lambda low, high, size: np.array([0, 1]))

    # Patch _save_checkpoint to do nothing and record call count.
    save_calls = []

    def fake_save_checkpoint(epoch):
        save_calls.append(epoch)

    monkeypatch.setattr(trainer, "_save_checkpoint", fake_save_checkpoint)

    # Call the train() method.
    loss_history = trainer.train()

    # With _steps = 3, we expect loss_history length = 3
    # and _save_checkpoint called 4 times (each iteration and final save).
    assert len(
        loss_history) == 3, (f"Expected 3 loss entries, "
                             f"got {len(loss_history)}.")
    assert all(cost == 0.42 for cost in
               loss_history), "Cost values are not as expected."
    # Expected parameters: initial_params * (0.9^3)
    expected_params = initial_params * (0.9 ** 3)
    np.testing.assert_allclose(trainer._parameters, expected_params, rtol=1e-5)
    # Check that _save_checkpoint was called 4 times.
    assert len(
        save_calls) == 4, \
        (f"Expected _save_checkpoint to be called 4 times, "
         f"got {len(save_calls)}.")


def test_train_returns_loss_history():
    """
    Test that train() returns a loss history of the correct length.
    """
    dataset = DummyQuantumDataset()
    trainer = ConcreteQTraining(number_of_qubits=4, dataset=dataset)
    trainer.timestamp = datetime.datetime.now()
    loss_history = trainer.train()
    assert isinstance(loss_history, list), "Loss history should be a list."
    assert len(
        loss_history) == trainer._steps, \
        "Loss history length does not match _steps."


def test_result():
    dataset = DummyQuantumDataset()
    # Use "cross entropy" to force predictions to be subscriptable.
    qtrainer = DummyQTraining(number_of_qubits=4, dataset=dataset,
                              cost_function="cross entropy")
    result = qtrainer.result()
    expected_keys = {"loss", "accuracy", "predictions", "labels", "gates",
                     "parameters", "date",
                     "single_check_accuracy", "triple_check_accuracy",
                     "quintil_check_accuracy"}
    missing = expected_keys - set(result.keys())
    assert not missing, f"Result is missing keys: {missing}"


def test_binominal_probability():
    prob = 0.8
    sample_size = 5
    bp = QTraining.binominal_probability(prob, sample_size)
    assert isinstance(bp,
                      float), "binominal_probability did not return a float."
    assert bp > 0, "binominal_probability should be positive."


def test_expectation_accuracy():
    predictions = [[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    labels = [1, 0, 1]
    acc_dict = QTraining.expectation_accuracy(predictions, labels)
    expected_keys = {"single_check_accuracy", "triple_check_accuracy",
                     "quintil_check_accuracy"}
    missing = expected_keys - set(acc_dict.keys())
    assert not missing, f"expectation_accuracy missing keys: {missing}"
    expected_single = (predictions[0][1] + predictions[1][0] + predictions[2][
        1]) / 3
    assert np.isclose(acc_dict["single_check_accuracy"], expected_single), \
        f"Expected single_check_accuracy {expected_single}, got {acc_dict['single_check_accuracy']}."


def test_init_from_save(tmp_path):
    dataset = DummyQuantumDataset()
    qtrainer = DummyQTraining(number_of_qubits=4, dataset=dataset)
    file_path = tmp_path / "checkpoint.pt"
    qtrainer.save(str(file_path))
    loaded_trainer = DummyQTraining.init_from_save(str(file_path))
    assert loaded_trainer.number_of_qubits == qtrainer.number_of_qubits, \
        "number_of_qubits not restored correctly."
    os.remove(str(file_path))


def test_load_save(monkeypatch, tmp_path):
    dataset = DummyQuantumDataset()
    qtrainer = DummyQTraining(number_of_qubits=4, dataset=dataset)
    dummy_checkpoint = {
        "parameters": np.array([0.1, 0.2, 0.3]),
        "optimizer_momentum": 0.95,
        "model_init_parameters": {"dummy": True},
        "loss_history": [],
        "number_of_qubits": 4,
        "dataset": dataset.convert_to_array(),
        "dataset_name": dataset.name,
        "optimizer": "nesterov",
        "embedding_type": "amplitude",
        "cost_function": "cross entropy"
    }
    monkeypatch.setattr(torch, "load", lambda file_location,
                                              weights_only=False: dummy_checkpoint)
    qtrainer.load("dummy_path")
    assert np.allclose(qtrainer._parameters, np.array([0.1, 0.2, 0.3]))
    assert qtrainer._opt.momentum == 0.95


if __name__ == "__main__":
    pytest.main()
