import os
import numpy as np
import pennylane.numpy as pnp
from typing import Optional
from qc_eval.training.training import Training


def test_square_loss():
    labels = [1, 0, 1]
    predictions = [0.9, 0.1, 0.8]
    expected_loss = sum(
        [(l - p) ** 2 for l, p in zip(labels, predictions)]) / len(labels)
    loss = Training._square_loss(labels, predictions)
    assert np.isclose(loss,
                      expected_loss), f"Expected {expected_loss}, got {loss}"


def test_cross_entropy():
    labels = [1, 0, 1]
    # For each pair, the loss is computed as:
    # c_entropy = l * log(p[l]) + (1 - l) * log(1 - p[1 - l])
    # then loss = -sum(c_entropy) / len[labels]
    predictions = [[0.9, 0.1], [0.7, 0.3], [0.8, 0.2]]
    expected_loss = - (1 * np.log(0.1) + 0 * np.log(0.9) +
                       0 * np.log(0.3) + 1 * np.log(0.7) +
                       1 * np.log(0.2) + 0 * np.log(0.8)) / 3
    loss = Training._cross_entropy(labels, predictions)
    assert np.isclose(loss,
                      expected_loss), f"Expected {expected_loss}, got {loss}"


def test_accuracy_test():
    # Dummy predictions: each prediction is a list [value1, value2]
    # Our implementation: prediction = 1 if pred[0] < pred[1] else 0.
    predictions = [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]]
    # This yields [1, 0, 1]. If we set labels to [1, 1, 1], accuracy = 2/3.
    labels = [1, 1, 1]
    accuracy = Training._accuracy_test(predictions, labels)
    expected_accuracy = 2 / 3
    assert np.isclose(accuracy,
                      expected_accuracy), f"Expected {expected_accuracy}, got {accuracy}"


def test_save_and_load_trainings_parameter(tmp_path):
    # Create a dummy tensor using pennylane.numpy (which is a thin wrapper around numpy)
    tensor = pnp.array([1, 2, 3])
    file_path = tmp_path / "params.npy"
    Training._save_trainings_parameter(tensor, str(file_path))
    loaded_tensor = Training._load_trainings_parameter(str(file_path))
    assert np.allclose(loaded_tensor,
                       tensor), "Loaded tensor does not match saved tensor"
    # Clean up file.
    os.remove(str(file_path))


# Create a dummy subclass of Training to test property methods.
class DummyTraining(Training):
    def __init__(self):
        self._parameters = {'param1': 1, 'param2': 2}
        self._neural_network = 'dummy_nn'
        self._loss_history = [0.1, 0.2]

    def train(self, finished_epoch: int = 0):
        return self._loss_history

    def _save_trainings_parameter(self, file_location: str):
        pass

    def _load_trainings_parameter(self, file_location: str):
        pass

    def save(self, file_location: str):
        pass

    def load(self, file_location: str):
        pass

    @classmethod
    def init_from_save(cls, file_location: str):
        return cls()

    def result(self):
        return {"dummy": True}

    def _create_checkpoint(self, epoch: Optional[int] = None) -> dict:
        return {"epoch": epoch}

    def _save_checkpoint(self, epoch: int) -> None:
        pass

    @classmethod
    def restart_training_from_checkpoint(cls,
                                         file_location: str) -> 'Training':
        return cls()


def test_property_methods():
    dummy = DummyTraining()
    assert dummy.parameters == {'param1': 1, 'param2': 2}
    assert dummy.neural_network == 'dummy_nn'
    # Check default property values from the abstract class.
    assert dummy.learning_rate == Training._learning_rate
    assert dummy.steps == Training._steps
    assert dummy.batch_size == Training._batch_size
    assert dummy.loss_history == [0.1, 0.2]


def test_default_property_values():
    # A second dummy subclass to test default values
    class DummyTrainingDefault(Training):
        def __init__(self):
            self._parameters = {}
            self._neural_network = None
            self._loss_history = []

        def train(self):
            pass

        def _save_trainings_parameter(self, file_location: str):
            pass

        def _load_trainings_parameter(self, file_location: str):
            pass

        def save(self, file_location: str):
            pass

        def load(self, file_location: str):
            pass

        def init_from_save(self, file_location: str):
            pass

        def result(self):
            pass

        def _create_checkpoint(self, epoch: Optional[int] = None) -> dict:
            return dict()

        def _save_checkpoint(self, epoch: int) -> None:
            pass

        @classmethod
        def restart_training_from_checkpoint(cls,
                                             file_location: str) -> 'Training':
            return cls()

    dummy_default = DummyTrainingDefault()
    assert dummy_default.learning_rate == Training._learning_rate
    assert dummy_default.steps == Training._steps
    assert dummy_default.batch_size == Training._batch_size
