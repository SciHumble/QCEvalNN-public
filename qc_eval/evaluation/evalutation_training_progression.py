import numpy as np
from qc_eval.evaluation import Evaluation
from qc_eval.evaluation.quantum_evaluation import QuantumEvaluation
from qc_eval.training import CTraining, QTraining
from qc_eval.training.training import TrainingParameters
from qc_eval.dataset import MNIST
from qc_eval.misc import notify
from qc_eval.evaluation.parameters_networks import *
from qc_eval.plotting import Scatter
from qc_eval import version_number
from matplotlib import pyplot as plt
from typing import Any
from datetime import datetime
from enum import Enum
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class EvaluationTrainingsProgression:
    relative_path: str = "../qc_eval/data/evaluation_trainings_progression/"
    absolute_path = os.path.abspath(relative_path)
    classical_file = absolute_path + f"/classical_results_{version_number}.csv"
    quantum_file = absolute_path + f"/quantum_results_{version_number}.csv"
    df: pd.DataFrame

    class ColumnEnum(Enum):
        model_name = "model_name"
        learning_rate = "learning_rate"
        batch_size = "batch_size"
        epochs = "epochs"
        accuracy = "accuracy"
        training_time = "training_time"
        flops = "flops"
        gates = "gates"
        parameters = "parameters"
        input_size = "input_size"
        layers = "layers"
        date = "date"
        optimizer = "optimizer"
        criterion = "criterion"
        dataset_name = "dataset_name"
        compacted_dataset = "compacted_dataset"
        feature_reduction = "feature_reduction"
        single_gate_error_rate = "single_gate_error_rate"
        cnot_error_rate = "cnot_error_rate"
        single_check_accuracy = "single_check_accuracy"
        triple_check_accuracy = "triple_check_accuracy"
        quintil_check_accuracy = "quintil_check_accuracy"
        predictions = "predictions"
        labels = "labels"
        convolution_layer = "convolution_layer"
        pooling_layer = "pooling_layer"

    def __init__(self, dataset: str, load_dataset=True):
        logger.debug(f"Init Evaluation with {locals()}.")
        self._init_dataframe()
        self.dataset_name = dataset
        if load_dataset:
            self._init_dataset()

    def _init_dataframe(self) -> None:
        if os.path.exists(self.classical_file):
            self.classical_df = pd.read_csv(self.classical_file)
        else:
            self.classical_df = pd.DataFrame(columns=[
                self.ColumnEnum.model_name.value,
                self.ColumnEnum.learning_rate.value,
                self.ColumnEnum.batch_size.value,
                self.ColumnEnum.epochs.value,
                self.ColumnEnum.accuracy.value,
                self.ColumnEnum.training_time.value,
                self.ColumnEnum.flops.value,
                self.ColumnEnum.parameters.value,
                self.ColumnEnum.input_size.value,
                self.ColumnEnum.layers.value,
                self.ColumnEnum.date.value,
                self.ColumnEnum.optimizer.value,
                self.ColumnEnum.criterion.value,
                self.ColumnEnum.dataset_name.value,
                self.ColumnEnum.compacted_dataset.value,
                self.ColumnEnum.feature_reduction.value,
            ])
        if os.path.exists(self.quantum_file):
            self.quantum_df = pd.read_csv(self.quantum_file)
        else:
            self.quantum_df = pd.DataFrame(columns=[
                self.ColumnEnum.model_name.value,
                self.ColumnEnum.learning_rate.value,
                self.ColumnEnum.batch_size.value,
                self.ColumnEnum.epochs.value,
                self.ColumnEnum.accuracy.value,
                self.ColumnEnum.training_time.value,
                self.ColumnEnum.gates.value,
                self.ColumnEnum.parameters.value,
                self.ColumnEnum.input_size.value,
                self.ColumnEnum.layers.value,
                self.ColumnEnum.date.value,
                self.ColumnEnum.optimizer.value,
                self.ColumnEnum.criterion.value,
                self.ColumnEnum.dataset_name.value,
                self.ColumnEnum.compacted_dataset.value,
                self.ColumnEnum.feature_reduction.value,
                self.ColumnEnum.single_gate_error_rate.value,
                self.ColumnEnum.cnot_error_rate.value,
                self.ColumnEnum.single_check_accuracy.value,
                self.ColumnEnum.triple_check_accuracy.value,
                self.ColumnEnum.quintil_check_accuracy.value,
                self.ColumnEnum.predictions.value,
                self.ColumnEnum.labels.value,
                self.ColumnEnum.convolution_layer.value,
                self.ColumnEnum.pooling_layer.value
            ])

    def _init_dataset(self, ) -> None:
        logger.debug("Init datasets.")

        def generate_data_dict(data_class: object()):
            reduction_numbers = set(NetworkParameters.input_numbers
                                    + NetworkParameters.qubit_numbers)
            pca = {n: data_class(f"pca{n}")
                   for n in reduction_numbers}
            pca_compact = {n: data_class(f"pca{n}", compact=True)
                           for n in reduction_numbers}
            autoencoder = {n: data_class(f"autoencoder{n}")
                           for n in reduction_numbers}
            autoencoder_compacted = {n: data_class(f"autoencoder{n}")
                                     for n in reduction_numbers}
            resize = {n: data_class(f"resize{int(2 ** n)}")
                      for n in NetworkParameters.qubit_numbers}
            return pca, pca_compact, autoencoder, autoencoder_compacted, resize

        name = self.dataset_name.lower()
        if name == "mnist":
            self.pca, self.pca_compact, self.autoencoder, self.autoencoder_compact, self.resize = generate_data_dict(
                MNIST)
        else:
            raise NotImplementedError(
                f"The dataset type {self.dataset_name!r} is not implemented."
            )

    def start(self, cnn=True, qcnn=True) -> None:
        logger.debug("Start evaluation process.")
        if cnn:
            classical_params = NetworkParameters.classical_params()
            for params in classical_params:
                input_num = params[ParamEnum.number_of_inputs.value]
                datasets = [self.pca[input_num],
                            self.autoencoder[input_num]]
                for dataset in datasets:
                    logger.debug(
                        f"Train neural network with parameters: {params}")
                    params["dataset"] = dataset
                    try:
                        classical_trainer = CTraining(**params)
                        start_time = self.timestamp
                        classical_trainer.train()
                        end_time = self.timestamp
                        result = classical_trainer.result()
                        self._store_classical_result(result,
                                                     end_time - start_time,
                                                     params)
                    except Exception as e:
                        logger.info(
                            f"Couldn't train cnn with parameters: {params}\n"
                            f"The reason is {e}."
                        )
            notify("Training CNN finished", "")
        if qcnn:
            quantum_params = NetworkParameters.quantum_params()
            count = 0
            for params in quantum_params:
                count += 1
                qubits = params[ParamEnum.number_of_qubits.value]
                notify(f"Start training QCNN "
                       f"({count}/{len(quantum_params)})",
                       f"The used parameters are {params!r}")
                logger.debug(f"Train neural network with parameters: {params}")
                try:
                    if params[
                        ParamEnum.embedding_type.value
                    ] == EmbeddingType.angle.value:
                        params["dataset"] = self.pca_compact[qubits]
                    elif params[
                        ParamEnum.embedding_type.value
                    ] == EmbeddingType.angle_compact.value:
                        params["dataset"] = self.pca_compact[qubits * 2]
                    elif params[
                        ParamEnum.embedding_type.value
                    ] == EmbeddingType.amplitude.value:
                        params["dataset"] = self.resize[qubits]
                    else:
                        raise NotImplementedError(
                            f"For the embedding "
                            f"{params[ParamEnum.embedding_type.value]!r} is "
                            f"not any dataset specified."
                        )
                    quantum_trainer = QTraining(**params)
                    finished_epochs = 0
                    for epochs in NetworkParameters.trainings_progression():
                        quantum_trainer._steps = epochs
                        start_time = self.timestamp
                        quantum_trainer.train(finished_epoch=finished_epochs)
                        end_time = self.timestamp
                        result = quantum_trainer.result()
                        self._store_quantum_result(result, end_time - start_time,
                                                   params)
                        finished_epochs = epochs
                        notify(f"Training up to epoch {epochs} is done.", "")

                    notify(f"Finished Training of QCNN ({count}"
                           f"/{len(quantum_params)})",
                           f"The QCNN with parameters {params!r}"
                           f"finished its training\n"
                           f"The results are: "
                           f"{self.quantum_notification_result(result)!r}")
                except Exception as e:
                    logger.info(
                        f"Couldn't train qcnn with parameters: {params}\n"
                        f"The reason is {e}."
                    )
                    notify("The training failed",
                           f"The training for the QCNN with the parameters "
                           f"{params!r} failed, because of {e!r}.")
            notify("Finished Training of all CNN and QCNN", "")

    def _store_classical_result(self, result: dict, trainings_time, params):
        new_row = pd.DataFrame([[
            result.get("model name", None),
            TrainingParameters.learning_rate.value,
            TrainingParameters.batch_size.value,
            TrainingParameters.steps.value,
            result.get("accuracy", None),
            trainings_time,
            result.get("number of flops", None),
            result.get("parameters", None),
            params.get(ParamEnum.number_of_inputs.value, None),
            params.get(ParamEnum.number_of_layers.value, None),
            result.get('date', None),
            params.get(ParamEnum.optimizer.value, "nesterov"),
            params.get(ParamEnum.cost_function.value, "cross entropy loss"),
            self.dataset_name,
            False,
            params["dataset"].feature_reduction if params.get("dataset",
                                                              None) is not None else None,
        ]], columns=self.classical_df.columns)
        self.classical_df = pd.concat([self.classical_df, new_row],
                                      ignore_index=True)
        self._save_df(self.classical_df, self.classical_file)

    def _store_quantum_result(self, result: dict, trainings_time,
                              params: dict):
        noise = {
            "single gate": params[ParamEnum.noise.value].get("single gate",
                                                             None) if params.get(
                ParamEnum.noise.value, None) is not None else None,
            "cnot": params[ParamEnum.noise.value].get("cnot",
                                                      None) if params.get(
                ParamEnum.noise.value, None) is not None else None}
        new_row = pd.DataFrame([[
            result.get("model name", None),
            TrainingParameters.learning_rate.value,
            TrainingParameters.batch_size.value,
            TrainingParameters.steps.value,
            result.get("accuracy", None),
            trainings_time,
            result.get("gates", None),
            result.get("parameters", None),
            params.get(ParamEnum.number_of_qubits.value, None),
            params.get(ParamEnum.number_of_layers.value, None),
            result.get('date', None),
            params.get(ParamEnum.optimizer.value, "nesterov"),
            params.get(ParamEnum.cost_function.value, "cross entropy loss"),
            self.dataset_name,
            True if params[
                        ParamEnum.embedding_type.value] == "angle" else False,
            params["dataset"].feature_reduction if params.get("dataset",
                                                              None) is not None else None,
            noise.get("single gate", None),
            noise.get("cnot", None),
            result.get("single_check_accuracy", None),
            result.get("triple_check_accuracy", None),
            result.get("quintil_check_accuracy", None),
            result.get("predictions", None),
            result.get("labels", None),
            params.get(ParamEnum.convolutional_circuit_number.value, None),
            params.get(ParamEnum.pooling_circuit_number.value, None)
        ]], columns=self.quantum_df.columns)
        self.quantum_df = pd.concat([self.quantum_df, new_row],
                                    ignore_index=True)
        self._save_df(self.quantum_df, self.quantum_file)

    @staticmethod
    def quantum_notification_result(result: dict) -> dict:
        value = {
            "Accuracy": result.get("accuracy", None),
            "Parameters": result.get("parameters", None),
            "Single Check Accuracy": result.get("single_check_accuracy", None),
            "Triple Check Accuracy": result.get("triple_check_accuracy", None),
            "Quintil Check Accuracy": result.get("quintil_check_accuracy",
                                                 None),
            "Average Prediction Probability": np.mean(np.array(
                QuantumEvaluation.expectation_probability(
                    result.get("predictions", []),
                    result.get("labels", []))
            ))
        }
        return value

    @staticmethod
    def prediction_probability(prediction, labels) -> list:
        value = [pred[0] if lab == 0 else pred[1]
                 for pred, lab in zip(prediction, labels)]
        return value

    @staticmethod
    def _save_df(df: pd.DataFrame, file_location):
        df.to_csv(file_location, index=False)

    def save(self, file_location: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, file_location: str) -> 'Evaluation':
        raise NotImplementedError

    def result(self, format_type: str = "text") -> Any:
        raise NotImplementedError

    def plot(self, x_data: str, y_data: str,
             z_data: str = None,
             plot_type: str = None,
             cnn_data: bool = True,
             output: str = None):
        """
        Options for data:
            'model_name', 'learning_rate', 'batch_size', 'epochs', 'accuracy',
            'training_time', 'gates', 'parameters', 'input_size', 'layers',
            'date', 'optimizer', 'criterion', 'dataset_name',
            'compacted_dataset', 'feature_reduction', 'single_gate_error_rate',
            'cnot_error_rate'
        Args:
            x_data:
            y_data:
            z_data:
            plot_type:
            cnn_data:
            output:

        Returns:

        """
        if cnn_data:
            params = {
                "x": self.classical_df[x_data],
                "x_axis": x_data,
                "y": self.classical_df[y_data],
                "y_axis": y_data
            }
            if z_data is not None:
                params.update({"z": self.classical_df[z_data],
                               "z_axis": z_data})
        else:
            params = {
                "x": self.quantum_df[x_data],
                "y": self.quantum_df[y_data]
            }
            if z_data is not None:
                params.update({"z": self.quantum_df[z_data],
                               "z_axis": z_data})

        if plot_type is None or plot_type.lower() == "scatter":
            plotter = Scatter(**params)
        else:
            raise NotImplementedError(
                f"The plot type {plot_type!r} is not implemented."
            )

        fig = plotter.plot()

        plt.show()

    def _generate_data_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def restart_from_autosave(cls, file_location):
        raise NotImplementedError

    def _autosave(self):
        raise NotImplementedError

    @property
    def timestamp(self):
        pass

    @timestamp.getter
    def timestamp(self) -> datetime:
        return datetime.now()
