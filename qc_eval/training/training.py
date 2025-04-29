from abc import ABC, abstractmethod
from typing import Any, Optional
from pennylane import numpy as pnp
from autograd import numpy as anp
from qc_eval.misc.parameters import TrainingParameters
import logging

logger = logging.getLogger(__name__)


class Training(ABC):
    _parameters: Any
    _neural_network: Any
    _learning_rate: float = TrainingParameters.learning_rate.value
    _steps: int = TrainingParameters.steps.value
    _batch_size: int = TrainingParameters.batch_size.value
    _save_rate: int = TrainingParameters.save_rate.value
    _loss_history: list

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, finished_epoch: int = 0) -> list:
        pass

    @staticmethod
    def _save_trainings_parameter(params: pnp.tensor,
                                  file_location: str):
        pnp.save(file_location, params)

    @staticmethod
    def _load_trainings_parameter(file_location: str) -> pnp.tensor:
        return pnp.load(file_location)

    @abstractmethod
    def save(self, file_location: str):
        pass

    @abstractmethod
    def load(self, file_location: str):
        pass

    @classmethod
    @abstractmethod
    def init_from_save(cls, file_location: str):
        pass

    @abstractmethod
    def result(self):
        pass

    @staticmethod
    def _square_loss(labels, predictions):
        logger.debug(
            f"Square loss called with labels {labels} and predictions "
            f"{predictions}."
        )
        loss = sum((l - p) ** 2 for l, p in zip(labels, predictions))
        loss /= len(labels)
        return loss

    @staticmethod
    def _cross_entropy(labels, predictions):
        logger.debug(
            f"Cross entropy called with labels {labels} and predictions "
            f"{predictions}."
        )
        loss = -sum(anp.log(p[l]) for p, l in zip(predictions, labels))
        loss /= len(labels)
        return loss

    @staticmethod
    def _accuracy_test(prediction, labels) -> float:
        prediction = [1 if pred[0] < pred[1] else 0 for pred in prediction]
        accuracy = sum(
            1 if pred == lab else 0 for pred, lab in zip(prediction, labels))
        accuracy /= len(labels)
        return accuracy

    @abstractmethod
    def _create_checkpoint(self,
                           epoch: Optional[int] = None) -> dict:
        pass

    @abstractmethod
    def _save_checkpoint(self, epoch: int) -> None:
        pass

    @classmethod
    @abstractmethod
    def restart_training_from_checkpoint(cls,
                                         file_location: str) -> 'Training':
        pass

    @property
    def parameters(self):
        pass

    @parameters.getter
    def parameters(self):
        return self._parameters

    @property
    def neural_network(self):
        pass

    @neural_network.getter
    def neural_network(self):
        return self._neural_network

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def loss_history(self) -> list:
        return self._loss_history
