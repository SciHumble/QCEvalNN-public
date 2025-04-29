import numpy as np
from qc_eval.training import CTraining, QTraining
from qc_eval.training.training import TrainingParameters
from qc_eval.dataset import MNIST
from qc_eval.misc import notify
from qc_eval.evaluation.parameters_networks import *
from qc_eval.evaluation.column_enum import ColumnEnum
from qc_eval.evaluation.quantum_evaluation import QuantumEvaluation
from qc_eval.plotting import Scatter
from qc_eval import version_number
from matplotlib import pyplot as plt
from typing import Any
from datetime import datetime
from enum import Enum
from pathlib import Path
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class Evaluation:
    path_to_data = Path(__file__).parent.parent / "data"
    classical_file = (path_to_data / "evaluation" /
                      f"classical_results_{version_number}.csv")
    quantum_file = (path_to_data / "evaluation" /
                    f"quantum_results_{version_number}.csv")
    df: pd.DataFrame
    ColumnEnum = ColumnEnum

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
                self.ColumnEnum.loss_history.value,
                self.ColumnEnum.number_of_features.value,
                self.ColumnEnum.autosafe_file.value,
                self.ColumnEnum.test_loss.value
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
                self.ColumnEnum.pooling_layer.value,
                self.ColumnEnum.loss_history.value,
                self.ColumnEnum.embedding.value,
                self.ColumnEnum.autosafe_file.value,
                self.ColumnEnum.test_loss.value
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
            (self.pca, self.pca_compact, self.autoencoder,
             self.autoencoder_compact, self.resize) = generate_data_dict(MNIST)
        else:
            raise NotImplementedError(
                f"The dataset type {self.dataset_name!r} is not implemented."
            )

    def start(self, ccnn=True, qcnn=True) -> None:
        logger.debug("Start evaluation process.")
        if qcnn:
            quantum_evaluator = QuantumEvaluation(
                self.dataset_name,
                self.pca,
                self.pca_compact,
                self.autoencoder,
                self.autoencoder_compact,
                self.resize,
                with_noise=True
            )
            quantum_evaluator.evaluate()

        if ccnn:
            classical_params = NetworkParameters.classical_params()
            for params in classical_params:
                input_num = params[ParamEnum.number_of_inputs.value]
                datasets = [self.pca_compact[input_num],
                            self.autoencoder_compact[input_num]]
                for dataset in datasets:
                    logger.debug(
                        f"Train neural network with parameters: {params}")
                    params["dataset"] = dataset
                    try:
                        classical_trainer = CTraining(**params)
                        start_time = self.timestamp
                        loss_history = classical_trainer.train()
                        end_time = self.timestamp
                        result = classical_trainer.result()
                        self._store_classical_result(
                            result,
                            end_time - start_time,
                            params, loss_history,
                            classical_trainer.autosafe_file
                        )
                    except Exception as e:
                        logger.info(
                            f"Couldn't train cnn with parameters: {params}\n"
                            f"The reason is {e}."
                        )
            notify("Training CNN finished", "")
            notify("Finished Training of all CNN and QCNN", "")

    def _store_classical_result(
            self, result: dict, trainings_time, params, loss_history: list,
            autosafe_file: str
    ):
        new_row = pd.DataFrame([[
            result.get("model name", None),
            TrainingParameters.learning_rate.value,
            TrainingParameters.batch_size.value,
            TrainingParameters.steps.value,
            result.get("accuracy", None),
            trainings_time,
            result.get("number of flops", None),
            result.get("number of parameters", None),
            params.get(ParamEnum.number_of_inputs.value, None),
            params.get(ParamEnum.number_of_layers.value, None),
            result.get('date', None),
            params.get(ParamEnum.optimizer.value, "nesterov"),
            params.get(ParamEnum.cost_function.value, "cross entropy loss"),
            self.dataset_name,
            False,
            params["dataset"].feature_reduction if params.get("dataset",
                                                              None) is not None else None,
            loss_history,
            params.get(ParamEnum.number_of_features.value),
            autosafe_file,
            result.get("test_loss", None)
        ]], columns=self.classical_df.columns)
        self.classical_df = pd.concat([self.classical_df, new_row],
                                      ignore_index=True)
        self._save_df(self.classical_df, self.classical_file)

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
                Evaluation.prediction_probability(
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
