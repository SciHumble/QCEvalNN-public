import torch
import torch.nn as nn
import logging
import numpy as np
from qc_eval.classical.convolutional_neural_network import ClassicalConvNN
from qc_eval.training.training import Training
from qc_eval.dataset.dataset import Dataset
from qc_eval.misc.parameters import TrainingParameters
from typing import Optional
from datetime import datetime
from qc_eval.misc.helper_functions import convert_datetime_to_string
from ptflops import get_model_complexity_info

logger = logging.getLogger(__name__)


class CTraining(Training):
    """
    Class to train and analyze the classical neural network

    Source:
        QCNN/CNN.py
    """
    momentum: float = TrainingParameters.momentum.value
    betas: tuple[float, float] = TrainingParameters.betas.value

    def __init__(self,
                 number_of_inputs: int,
                 dataset: Dataset,
                 number_of_layers: Optional[int] = None,
                 optimizer: str = "nesterov",
                 number_of_features: Optional[int] = None,
                 parameters: Optional[np.ndarray] = None,
                 **kwargs):
        logger.debug(f"Init CTraining with {locals()}.")
        self.number_of_inputs: int = number_of_inputs
        self._cnn_init_parameters = {"sequence_length": self.number_of_inputs,
                                     "num_features": number_of_features,
                                     "num_layers": number_of_layers}
        self._neural_network = ClassicalConvNN(**self._cnn_init_parameters)
        self.cnn = self._neural_network.ccnn
        self.dataset: Dataset = dataset
        self._optimizer_name = optimizer.lower()
        self._init_optimizer()
        self._parameters = parameters or np.random.randn(
            self.neural_network.num_parameters)
        self.criterion = nn.CrossEntropyLoss()
        self._loss_history = []
        self.timestamp = datetime.now()
        self.autosafe_file = (f"cnn-{self.number_of_inputs}"
                              f"-{self.neural_network.num_layers}"
                              f"-{convert_datetime_to_string(self.timestamp)}"
                              f".pt")
        self.additional_attributes = kwargs

    def _init_optimizer(self):
        if self._optimizer_name.lower() == "nesterov":
            self._opt = torch.optim.SGD(self.cnn.parameters(),
                                        lr=self.learning_rate,
                                        momentum=self.momentum,
                                        nesterov=True)
        elif self._optimizer_name.lower() == "adam":
            self._opt = torch.optim.Adam(self.cnn.parameters(),
                                         lr=self.learning_rate,
                                         betas=self.betas)
        else:
            raise NotImplementedError(
                f"The optimizer {self._optimizer_name!r} is not implement."
                f" There are 'Nesterov' and 'Adam' as optimizer."
            )

    def _create_checkpoint(self,
                           epoch: Optional[int] = None) -> dict:
        logger.debug("Creating checkpoint.")
        checkpoint = {
            "number_of_inputs": self.number_of_inputs,
            "dataset": self.dataset.convert_to_array(),
            "model_init_parameters": self._cnn_init_parameters,
            "model_state_dict": self._neural_network.state_dict,
            "parameters": self._parameters,
            "optimizer_state_dict": self._opt.state_dict,
            "optimizer": self._optimizer_name,
            "loss_history": self.loss_history,
        }
        if epoch is not None:
            checkpoint["epoch"] = epoch
        return checkpoint

    def train(self, finished_epoch: int = 0) -> list:
        logger.info("Start training.")
        x_train, x_test, y_train, y_test = self.dataset.dataset()

        for it in range(finished_epoch, self.steps):
            batch_index = np.random.randint(0, len(x_train),
                                            (self.batch_size,))
            x_batch = np.array([x_train[i] for i in batch_index])
            y_batch = np.array([y_train[i] for i in batch_index])

            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            x_batch.resize_(self.batch_size, 1, self.number_of_inputs)
            y_batch = torch.tensor(y_batch, dtype=torch.long)

            self.criterion = nn.CrossEntropyLoss()
            self._init_optimizer()

            prediction = self.cnn(x_batch)

            loss = self.criterion(prediction, y_batch)
            self._loss_history.append(loss.item())

            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

            if it % self._save_rate == 0:
                logger.info(
                    f"Trainingsiteration: {it}/{self.steps}\t\t"
                    f"Loss: {np.round(loss.item(), 6)}\t\t"
                )
                self._save_checkpoint(it)

        logger.debug(
            "Finished training neural network."
        )
        return self.loss_history

    def get_flops(self):
        """
        Source:
            https://pypi.org/project/ptflops/
        Returns:
            FLOPs
        """
        input_shape = (1, self.number_of_inputs,)
        macs, params = get_model_complexity_info(
            self.cnn,
            input_shape,
            as_strings=False,
            backend='pytorch',
            print_per_layer_stat=False,
            verbose=True
        )
        # MACs: Multiply-Accumulates
        # FLOPs: Floating Point Operation
        # flops = 2 * macs
        flops = 2 * macs
        return flops

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
    def init_from_save(cls, file_location: str) -> 'CTraining':
        checkpoint = torch.load(file_location)

        instance = cls.__new__(cls)
        instance.number_of_inputs = checkpoint["number_of_inputs"]
        instance.dataset = CTraining.restore_dataset(checkpoint)
        instance._cnn_init_parameters = checkpoint["model_init_parameters"]
        instance._neural_network = ClassicalConvNN(
            **instance._cnn_init_parameters
        )
        instance._neural_network.load_state_dict(
            checkpoint["model_state_dict"])
        instance._parameters = checkpoint["parameters"]
        instance._init_optimizer()
        instance._opt.load_state_dict(checkpoint["optimizer_state_dict"])
        instance.timestamp = datetime.now()
        instance.criterion = nn.CrossEntropyLoss()
        instance._loss_history = checkpoint["loss_history"]

        logger.info(f"Model and parameters initialized from {file_location}.")
        return instance

    @classmethod
    def restart_training_from_checkpoint(cls,
                                         file_location: str) -> 'CTraining':
        instance = cls.init_from_save(file_location)
        checkpoint = torch.load(file_location)
        instance.train(checkpoint["epoch"])
        return instance

    def load(self, file_location: str) -> None:
        checkpoint = torch.load(file_location)
        self._neural_network.load_state_dict(checkpoint['model_state_dict'])
        self._opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self._parameters = checkpoint['parameters']
        logger.info(f"Model loaded from {file_location}.")

    def save(self, file_location: str) -> None:
        state_dict = self._create_checkpoint()
        torch.save(state_dict, file_location)
        logger.info(f"Model saved to {file_location}.")

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint = self._create_checkpoint(epoch)
        file_location = (TrainingParameters.autosafe_folder.value
                         / self.autosafe_file)
        torch.save(checkpoint, file_location)

    def result(self) -> dict:
        _, x_test, _, y_test = self.dataset.dataset()
        number_of_test = len(x_test)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        x_test.resize_(number_of_test, 1, self.number_of_inputs)
        y_test = torch.tensor(y_test, dtype=torch.long)

        self.neural_network.eval()
        with torch.no_grad():
            predictions = self.cnn(x_test)

        test_loss = self.criterion(predictions, y_test)

        predictions = predictions.detach().numpy()

        predicted_labels = [1 if pred[0] >= pred[1] else 0 for pred in
                            predictions]
        accuracy = self._accuracy_test(predictions,
                                       y_test.numpy())
        logger.info(f"Model accuracy on test set: {accuracy:.4f}")

        return {
            'accuracy': accuracy,
            'predictions': predicted_labels,
            'parameters': self.neural_network.parameters,
            'number of parameters': self.neural_network.num_parameters,
            'number of flops': self.get_flops(),
            'model name': "cnn",
            'date': self.timestamp,
            'test_loss': test_loss
        }

    @property
    def number_of_parameter(self) -> int:
        pass

    @number_of_parameter.getter
    def number_of_parameter(self) -> int:
        number_of_parameter = sum(
            p.nelement for p in list(self.cnn.parameters()))
        return number_of_parameter


if __name__ == "__main__":
    from qc_eval.dataset import MNIST
    trainer = CTraining(4, MNIST("pca4"), 1, "adam", 3)
    trainer.train()
    print(trainer.result())
