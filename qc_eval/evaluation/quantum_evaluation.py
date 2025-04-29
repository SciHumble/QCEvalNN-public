from qc_eval.training.training import TrainingParameters
import pandas as pd
import numpy as np
from qc_eval.training import QTraining
from qc_eval.evaluation.parameters_networks import *
from qc_eval.evaluation.column_enum import ColumnEnum
from datetime import datetime
import logging
from queue import Queue
from pathlib import Path
import json
import os
from qc_eval.misc import notify
from qc_eval import version_number


logger = logging.getLogger(__name__)

"""
This script is for the testing and evaluation of QCNN over the whole parameter
realm.
The QCNNs should get evaluated in the first part without noise and in the
second part with the known parameters tested with noise.
This has the purpose of reducing calculation time.
"""


class QuantumEvaluation:
    path_to_queue = (Path(__file__).parent.parent / "data" / "evaluation" /
                     "quantum_evaluation_queue.json")
    path_to_queue_noise = (Path(__file__).parent.parent / "data" /
                           "evaluation" /
                           "quantum_evaluation_queue_noise.json")
    path_to_dataframe = (Path(__file__).parent.parent / "data" / "evaluation" /
                         f"quantum_results_{version_number}.csv")
    path_to_queue_failed = (Path(__file__).parent.parent / "data" /
                            "evaluation" /
                            f"quantum_evaluation_queue_failed.json")

    def __init__(self, dataset_name: str,
                 pca: dict, pca_compact: dict,
                 autoencoder: dict, autoencoder_compact: dict,
                 resize: dict, with_noise: bool = True,
                 skip_load_queues: bool = False):
        self.evaluation_parameters = NetworkParameters.quantum_params()
        self.queue = Queue()
        self.queue_noise = Queue()
        self.failed_tasks = Queue()
        if not skip_load_queues:
            self._load_queues()
        self.pca, self.pca_compact = pca, pca_compact
        self.autoencoder = autoencoder
        self.autoencoder_compact = autoencoder_compact
        self.resize = resize
        self.dataset_name = dataset_name
        self._init_dataframe()
        self.with_noise = with_noise

    def _load_queues(self):
        """Load tasks from a file into the queue."""
        try:
            with open(self.path_to_queue, 'r') as f:
                tasks = json.load(f)
                for task in tasks:
                    self.queue.put(task)
        except FileNotFoundError:
            for params in self.evaluation_parameters:
                self.queue.put(params)

        try:
            with open(self.path_to_queue_noise, "r") as f:
                tasks = json.load(f)
                for task in tasks:
                    self.queue_noise.put(task)
        except FileNotFoundError:
            pass

        try:
            with open(self.path_to_queue_failed, "r") as f:
                tasks = json.load(f)
                for task in tasks:
                    self.failed_tasks.put(task)
        except FileNotFoundError:
            pass

    def _store_queues(self):
        if not self.queue.empty():
            with open(self.path_to_queue, 'w') as f:
                tasks = list(self.queue.queue)
                json.dump(tasks, f)
        if not self.queue_noise.empty():
            with open(self.path_to_queue_noise, "w") as f:
                tasks = list(self.queue_noise.queue)
                json.dump(tasks, f)
        if not self.failed_tasks.empty():
            with open(self.path_to_queue_failed, "w") as f:
                tasks = list(self.failed_tasks.queue)
                json.dump(tasks, f)

    def _init_dataframe(self):
        if os.path.exists(self.path_to_dataframe):
            self.quantum_df = pd.read_csv(self.path_to_dataframe)
        else:
            self.quantum_df = pd.DataFrame(columns=[
                ColumnEnum.model_name.value,
                ColumnEnum.learning_rate.value,
                ColumnEnum.batch_size.value,
                ColumnEnum.epochs.value,
                ColumnEnum.accuracy.value,
                ColumnEnum.training_time.value,
                ColumnEnum.gates.value,
                ColumnEnum.parameters.value,
                ColumnEnum.input_size.value,
                ColumnEnum.layers.value,
                ColumnEnum.date.value,
                ColumnEnum.optimizer.value,
                ColumnEnum.criterion.value,
                ColumnEnum.dataset_name.value,
                ColumnEnum.compacted_dataset.value,
                ColumnEnum.feature_reduction.value,
                ColumnEnum.single_gate_error_rate.value,
                ColumnEnum.cnot_error_rate.value,
                ColumnEnum.single_check_accuracy.value,
                ColumnEnum.triple_check_accuracy.value,
                ColumnEnum.quintil_check_accuracy.value,
                ColumnEnum.predictions.value,
                ColumnEnum.labels.value,
                ColumnEnum.convolution_layer.value,
                ColumnEnum.pooling_layer.value,
                ColumnEnum.loss_history.value,
                ColumnEnum.embedding.value,
                ColumnEnum.autosafe_file.value
            ])

    def _eval_noise_free(self):
        count = 0
        total_count = self.queue.qsize()
        while not self.queue.empty():
            task = self.queue.get()
            count += 1
            notify(f"Start training QCNN "
                   f"({count}/{total_count})",
                   f"The used parameters are {task!r}")
            logger.debug(f"Train neural network with parameters: {task}")

            try:
                result, parameters = self._train_and_eval_noise_free(task)
                notify(f"Finished Training of QCNN ({count}"
                       f"/{total_count})",
                       f"The QCNN with parameters {task!r}"
                       f"finished its training\n"
                       f"The results are: "
                       f"{self.quantum_notification_result(result)!r}")
                self.queue.task_done()
                self._add_task_to_queue_noise(task, parameters)
            except Exception as e:
                logger.info(
                    f"Couldn't train qcnn with parameters: {task}\n"
                    f"The reason is {e}."
                )
                notify("The training failed",
                       f"The training for the QCNN with the parameters "
                       f"{task!r} failed, because of {e!r}.")
                task.pop("dataset")
                self.failed_tasks.put(task)
                self.queue.task_done()

        # queue is empty -> queue file can be deleted
        self.path_to_queue.unlink(missing_ok=True)

    def _train_and_eval_noise_free(self, params):
        qubits = params[ParamEnum.number_of_qubits.value]
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
        start_time = self.timestamp
        loss_history = quantum_trainer.train()
        end_time = self.timestamp
        result = quantum_trainer.result()
        parameters = quantum_trainer.parameters
        self._store_quantum_result(
            result, end_time - start_time, params, loss_history,
            quantum_trainer.autosafe_file
        )
        return result, parameters

    def _add_task_to_queue_noise(self, task: dict, parameters) -> None:
        """
        Adds the finished noise-free trained qcnn to the noisy queue for later
        testing the parameters unter a noisy environment.
        """
        task_template = task.copy()
        task_template.pop("dataset")
        task_template.update({"parameters": parameters})

        controlled = NetworkParameters.noise_testing["controlled"]
        single = NetworkParameters.noise_testing["single"]
        if len(controlled) != len(single):
            raise ValueError(
                f"The number control gate and single gate depolarization error"
                f" must be same."
            )
        noise_pairs = [{"cnot": c,
                        "single gate": s} for c, s in zip(controlled, single)]

        for noise in noise_pairs:
            task_template.update({ParamEnum.noise.value: noise})
            # must be a copy because later the parameters gets delete,
            # but this should only happen for the current task
            self.queue_noise.put(task_template.copy())

    def _eval_with_noise(self):
        count = 0
        total_count = self.queue_noise.qsize()
        while not self.queue_noise.empty():
            task = self.queue_noise.get()
            parameters = task["parameters"]
            del task["parameters"]
            count += 1
            notify(f"Start testing QCNN with noise "
                   f"({count}/{total_count})",
                   f"The used parameters are {task!r}")
            logger.debug(f"Train neural network with parameters: {task}")

            try:
                result = self._test_with_noise(task, parameters)
                notify(f"Finished Training of QCNN ({count}"
                       f"/{total_count})",
                       f"The QCNN with parameters {task!r}"
                       f"finished its training\n"
                       f"The results are: "
                       f"{self.quantum_notification_result(result)!r}")
                self.queue_noise.task_done()
            except Exception as e:
                logger.info(
                    f"Couldn't train qcnn with parameters: {task}\n"
                    f"The reason is {e}."
                )
                notify("The training failed",
                       f"The training for the QCNN with the parameters "
                       f"{task!r} failed, because of {e!r}.")
                task.update({"parameters": parameters})
                task.pop("dataset")
                self.failed_tasks.put(task)
                self.queue_noise.task_done()

        # queue is empty -> queue file can be deleted
        self.path_to_queue_noise.unlink(missing_ok=True)

    def _test_with_noise(self, task, params):
        qubits = task[ParamEnum.number_of_qubits.value]
        if task[
            ParamEnum.embedding_type.value
        ] == EmbeddingType.angle.value:
            task["dataset"] = self.pca_compact[qubits]
        elif task[
            ParamEnum.embedding_type.value
        ] == EmbeddingType.angle_compact.value:
            task["dataset"] = self.pca_compact[qubits * 2]
        elif task[
            ParamEnum.embedding_type.value
        ] == EmbeddingType.amplitude.value:
            task["dataset"] = self.resize[qubits]
        else:
            raise NotImplementedError(
                f"For the embedding "
                f"{task[ParamEnum.embedding_type.value]!r} is "
                f"not any dataset specified."
            )

        quantum_trainer = QTraining(**task)
        quantum_trainer._parameters = params
        start_time = self.timestamp
        result = quantum_trainer.result()
        end_time = self.timestamp
        self._store_quantum_result(
            result, end_time - start_time, task, [],
            autosafe_file=quantum_trainer.autosafe_file
        )
        return result

    def _store_quantum_result(
            self, result: dict, trainings_time, params: dict,
            loss_history: list, autosafe_file: str
    ) -> None:
        noise = {
            "single gate": params[ParamEnum.noise.value].get("single gate",
                                                             None) if params.get(
                ParamEnum.noise.value, None) is not None else None,
            "cnot": params[ParamEnum.noise.value].get("cnot",
                                                      None) if params.get(
                ParamEnum.noise.value, None) is not None else None}

        feature_reduction = None
        if params.get("dataset", None) is not None:
            dataset = params["dataset"]
            if hasattr(dataset, "feature_reduction"):
                feature_reduction = params["dataset"].feature_reduction

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
            feature_reduction,
            noise.get("single gate", None),
            noise.get("cnot", None),
            result.get("single_check_accuracy", None),
            result.get("triple_check_accuracy", None),
            result.get("quintil_check_accuracy", None),
            result.get("predictions", None),
            result.get("labels", None),
            params.get(ParamEnum.convolutional_circuit_number.value, None),
            params.get(ParamEnum.pooling_circuit_number.value, None),
            loss_history,
            params.get(ParamEnum.embedding_type.value, None),
            autosafe_file,
            result.get("loss")
        ]], columns=self.quantum_df.columns)
        self.quantum_df = pd.concat([self.quantum_df, new_row],
                                    ignore_index=True)
        self._save_df(self.quantum_df, self.path_to_dataframe)

    def evaluate(self):
        try:
            self._eval_noise_free()
            if self.with_noise:
                self._eval_with_noise()
        except Exception as e:
            raise e
        finally:
            self._store_queues()

    @staticmethod
    def _save_df(dataframe: pd.DataFrame, path):
        dataframe.to_csv(path, index=False)

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
    def expectation_probability(predictions, labels) -> list:
        value = [pred[0] if lab == 0 else pred[1]
                 for pred, lab in zip(predictions, labels)]
        return value

    @property
    def timestamp(self):
        pass

    @timestamp.getter
    def timestamp(self) -> datetime:
        return datetime.now()
