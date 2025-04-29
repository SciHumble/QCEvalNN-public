import torch
from datetime import datetime
from qc_eval.training.training import Training
from qc_eval.quantum import QCNN
from qc_eval.misc.parameters import QuantumOptimizer, TrainingParameters
from typing import Optional
from qc_eval.dataset.dataset import Dataset
from qc_eval.misc.helper_functions import convert_datetime_to_string
from pennylane import numpy as pnp
import numpy as np
import pennylane as qml
from scipy.special import binom
import logging

logger = logging.getLogger(__name__)


def generate_noise_dict(cnot_error_rate: float = 0.0,
                        single_gate_error_rate: float = 0.0) -> dict:
    """
    Generates the dictionary for adding noise into the qcnn.

    Args:
        cnot_error_rate: Error rate for a CNOT gate
        single_gate_error_rate: Error rate for a single gate

    Returns:
        Dictionary with the error rates
    """
    noise_dict = {"cnot": cnot_error_rate,
                  "single gate": single_gate_error_rate}
    return noise_dict


class QTraining(Training):
    """
    Class to train quantum neural network. In the current version it trains
    a QCNN.
    """

    def __init__(self,
                 number_of_qubits: int,
                 dataset: Dataset,
                 convolutional_circuit_number: Optional[int] = None,
                 pooling_circuit_number: Optional[int] = None,
                 embedding_type: str = "amplitude",
                 cost_function: str = "cross entropy",
                 parameters: Optional[np.ndarray] = None,
                 noise: Optional[dict] = None,
                 **kwargs):
        logger.debug(f"Init QTraining with {locals()}.")
        self.number_of_qubits = number_of_qubits
        if noise is None:
            self.error_rate = None
        else:
            self.error_rate = noise
        self._qcnn_init_parameters = {
            "num_qubits": number_of_qubits,
            "num_layers": None,
            "conv_circ": convolutional_circuit_number,
            "pool_circ": pooling_circuit_number,
            "noise": self.error_rate
        }
        self.quantum_circuit = QCNN(**self._qcnn_init_parameters)
        self._neural_network = self.quantum_circuit.circuit
        self._optimizer_name = "nesterov"
        qml.QNGOptimizer(stepsize=self.learning_rate)
        self._opt = qml.NesterovMomentumOptimizer(stepsize=self.learning_rate)
        self.dataset = dataset
        self.embedding_type = embedding_type
        self.cost_function = cost_function
        if parameters is None:
            self._parameters = pnp.random.randn(
                self.quantum_circuit.num_parameters
            )
        else:
            self._parameters = parameters
        self._loss_history = []
        self.timestamp = datetime.now()
        self.autosafe_file = self.generate_autosafe_file_name(
            self.quantum_circuit.num_qubits, self.timestamp
        )
        self.additional_attributes = kwargs

    def _get_optimizer(self):
        if self._optimizer_name == QuantumOptimizer.nesterov.value:
            return qml.NesterovMomentumOptimizer(stepsize=self.learning_rate)
        else:
            raise ValueError(
                f"The optimizer {self._optimizer_name!r} is unknown, please "
                f"choose one out of qc_eval.misc.parameters.QuantumOptimizer."
            )

    def _create_checkpoint(self, epoch: Optional[int] = None) -> dict:
        logger.debug(f"Creating checkpoint.")
        checkpoint = {
            "number_of_qubits": self.quantum_circuit.num_qubits,
            "dataset": self.dataset.convert_to_array(),
            "dataset_name": self.dataset.name,
            "model_init_parameters": self._qcnn_init_parameters,
            "parameters": self._parameters,
            "optimizer_momentum": self._opt.momentum,
            "optimizer": self._optimizer_name,
            "embedding_type": self.embedding_type,
            "cost_function": self.cost_function,
            "loss_history": self.loss_history,
        }
        if epoch is not None:
            checkpoint["epoch"] = epoch
        return checkpoint

    def train(self, finished_epoch: int = 0) -> list:
        logger.info("Start training.")
        x_train, x_test, y_train, y_test = self.dataset.dataset()

        for it in range(finished_epoch, self.steps):
            batch_index = pnp.random.randint(0, len(x_train),
                                             (self.batch_size,))
            x_batch = [x_train[i] for i in batch_index]
            y_batch = [y_train[i] for i in batch_index]
            cost = lambda v: self._cost(v, x_batch, y_batch,
                                        self.embedding_type,
                                        self.cost_function)
            self._parameters, cost_new = self._opt.step_and_cost(
                cost,
                self._parameters
            )
            self._loss_history.append(cost_new)
            if it % self._save_rate == 0:
                logger.info(
                    f"Trainingsiteration: {it}/{self.steps}\t"
                    f"Loss: {np.round(cost_new, 6)}")
                self._save_checkpoint(it)
        logger.info("Finished Training")
        self._save_checkpoint(self.steps-1)
        return self.loss_history

    def _cost(self, params, x_data, y,
              embedding_type: str,
              cost_function: str = 'cross entropy'):
        logger.debug(f"Starting calculating the cost/loss.")
        predictions = [self.quantum_circuit.qnode(x, params, embedding_type,
                                                  cost_function) for
                       x in x_data]
        if cost_function == "mse":
            loss = self._square_loss(y, predictions)
        elif cost_function == "cross entropy":
            loss = self._cross_entropy(y, predictions)
        else:
            raise NotImplementedError(
                'Currently are only the cost functions "mse" and '
                '"cross entropy" implemented.'
            )
        logger.debug(f"Finished calculating the cost/loss: {loss}.")
        return loss

    @staticmethod
    def restore_dataset(checkpoint) -> Dataset:
        from qc_eval.dataset import MNIST
        name = checkpoint.get("dataset_name", MNIST.name)

        if name == MNIST.name:
            dataset = MNIST.restore_from_array(checkpoint["dataset"])
        else:
            raise NotImplementedError(
                f"The dataset {name!r} is not yet implemented."
            )

        return dataset

    @classmethod
    def init_from_save(cls, file_location: str) -> 'QTraining':
        logger.debug(
            f"Initializing QTraining from the save file {file_location!r}"
        )
        checkpoint = torch.load(file_location, weights_only=False)

        instance = cls.__new__(cls)
        instance._qcnn_init_parameters = checkpoint["model_init_parameters"]
        instance.quantum_circuit = QCNN(**instance._qcnn_init_parameters)
        instance._neural_network = instance.quantum_circuit.circuit
        instance.dataset = QTraining.restore_dataset(checkpoint)
        instance._optimizer_name = checkpoint["optimizer"]
        if instance._optimizer_name.lower() == "nesterov":
            instance._opt = qml.NesterovMomentumOptimizer(
                stepsize=instance.learning_rate)
        else:
            raise NotImplementedError(
                f"The optimizer {instance._optimizer_name} is not implemented."
            )
        instance._opt.momentum = checkpoint["optimizer_momentum"]
        instance.embedding_type = checkpoint["embedding_type"]
        instance.cost_function = checkpoint["cost_function"]
        instance._parameters = checkpoint["parameters"]
        instance._loss_history = checkpoint["loss_history"]
        instance.timestamp = datetime.now()
        instance.autosafe_file = instance.generate_autosafe_file_name(
            instance.quantum_circuit.num_qubits,
            instance.timestamp
        )

        logger.info(f"Quantum model initialized from {file_location}.")
        return instance

    def load(self, file_location: str) -> None:
        logger.debug(f"Loading model state from file {file_location!r}.")
        checkpoint = torch.load(file_location)
        self._parameters = checkpoint["parameters"]
        self._opt.momentum = checkpoint["optimizer_momentum"]

    def save(self, file_location: str,
             loss_history: Optional[list] = None) -> None:
        logger.debug(f"Saving model state to file {file_location!r}")
        checkpoint = self._create_checkpoint(loss_history=loss_history)
        pnp.save(file_location, checkpoint)
        logger.info(f"Quantum model saved to {file_location}.")

    def _save_checkpoint(self, epoch: int) -> None:
        logger.debug(f"Saving model checkpoint at epoch {epoch}.")
        checkpoint = self._create_checkpoint(epoch)
        file_location = f"qcnn-{self.quantum_circuit.num_qubits}-"
        file_location += convert_datetime_to_string(self.timestamp)
        file_location += ".pt"
        file_location = (TrainingParameters.autosafe_folder.value
                         / file_location)
        torch.save(checkpoint, file_location)

    @classmethod
    def restart_training_from_checkpoint(cls,
                                         file_location: str) -> 'Training':
        instance = cls.init_from_save(file_location)
        checkpoint = torch.load(file_location)
        logger.debug(f"Restart training at epoch {checkpoint['epoch']}.")
        instance.train(checkpoint["epoch"])
        return instance

    def result(self) -> dict:
        logger.debug(f"Evaluate the neural network and generating the result.")
        _, x_test, _, y_test = self.dataset.dataset()
        predictions = [self.quantum_circuit.qnode(x, self._parameters,
                                                  self.embedding_type,
                                                  self.cost_function) for x in
                       x_test]

        if self.cost_function == "mse":
            loss = self._square_loss(y_test, predictions)
        elif self.cost_function == "cross entropy":
            loss = self._cross_entropy(y_test, predictions)
        else:
            raise NotImplementedError(
                'Currently, only the cost functions "mse" and "cross entropy" '
                'are implemented.'
            )

        accuracy = self._accuracy_test(predictions, y_test)

        logger.info(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")

        value = {'loss': loss,
                 'accuracy': accuracy,
                 'predictions': predictions,
                 'labels': list(y_test),
                 'gates': self.quantum_circuit.gates,
                 'parameters': self.quantum_circuit.num_parameters,
                 'date': self.timestamp}

        value.update(self.expectation_accuracy(predictions, y_test))

        return value

    @staticmethod
    def generate_autosafe_file_name(qubits: int, timestamp: datetime) -> str:
        autosafe_file = (f"qcnn-{qubits}"
                         f"-{convert_datetime_to_string(timestamp)}.pt")
        return autosafe_file

    @staticmethod
    def binominal_probability(probability: float, sample_size: int) -> float:
        """
        Calculates the binominal probability of measuring a state, when there
        is a fixed sample size and want to know how often the measured majority
        is this state.

        Args:
            probability: probability of measuring this state
            sample_size: number of measurements done

        Returns:
            probability of measuring the state in the majority of times
        """
        def binominal():
            val = binom(sample_size, k)
            val *= probability**k
            val *= (1-probability)**(sample_size-k)
            return val
        value = 0.
        for k in range(sample_size//2+1, sample_size+1):
            value += binominal()
        return value

    @staticmethod
    def expectation_accuracy(predictions: list[list], labels) -> dict:
        def single_check():
            value = sum(
                pred[lab]
                for pred, lab in zip(predictions, labels[:len(predictions)])
            )/len(predictions)
            return value

        def triple_check():
            value = sum(
                QTraining.binominal_probability(prob[lab], 3)
                for prob, lab in zip(predictions, labels[:len(predictions)])
            )/len(predictions)
            return value

        def quintil_check():
            value = sum(
                QTraining.binominal_probability(prob[lab], 5)
                for prob, lab in zip(predictions, labels[:len(predictions)])
            ) / len(predictions)
            return value

        val = {
            "single_check_accuracy": single_check(),
            "triple_check_accuracy": triple_check(),
            "quintil_check_accuracy": quintil_check(),
        }

        return val

    @property
    def neural_network(self):
        pass

    @neural_network.getter
    def neural_network(self):
        return self._neural_network


if __name__ == "__main__":
    from qc_eval.dataset import MNIST
    from qc_eval.misc.parameters import EmbeddingType

    dataset = MNIST("pca4")
    quantum_trainer = QTraining(
        number_of_qubits=4,
        dataset=dataset,
        convolutional_circuit_number=1,
        pooling_circuit_number=2,
        embedding_type=EmbeddingType.angle.value
    )
    quantum_trainer.train()
    print(quantum_trainer.result())
