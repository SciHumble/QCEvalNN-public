import os
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from queue import Queue
from qc_eval.evaluation.quantum_evaluation import QuantumEvaluation
from qc_eval.evaluation.parameters_networks import NetworkParameters, \
    ParamEnum, EmbeddingType


# Fixture: Provide dummy dataset objects for evaluation.
@pytest.fixture
def dummy_dataset_objects() -> dict:
    dummy = MagicMock()
    dummy.feature_reduction = "dummy"
    return {
        "pca": {4: dummy},
        "pca_compact": {4: dummy},
        "autoencoder": {4: dummy},
        "autoencoder_compact": {4: dummy},
        "resize": {4: dummy}
    }


# Fixture: Provide dummy evaluation parameters.
@pytest.fixture
def mock_evaluation_parameters() -> list:
    params = [{
        ParamEnum.number_of_qubits.value: 4,
        ParamEnum.convolutional_circuit_number.value: None,
        ParamEnum.pooling_circuit_number.value: None,
        ParamEnum.embedding_type.value: EmbeddingType.angle.value,
        ParamEnum.cost_function.value: "cross entropy",
        ParamEnum.noise.value: None,
        ParamEnum.number_repetition.value: 1
    }]
    return params


# Fixture: Create a QuantumEvaluation instance with patched methods.
@pytest.fixture
def mock_quantum_evaluator(monkeypatch, dummy_dataset_objects,
                           mock_evaluation_parameters):
    # Override _init_dataframe to quickly set an empty DataFrame.
    monkeypatch.setattr(QuantumEvaluation, "_init_dataframe",
                        lambda self: setattr(self, "quantum_df",
                                             pd.DataFrame()))

    # Patch _load_queues to load tasks from our dummy evaluation parameters.
    def fake_load_queues(self):
        for params in mock_evaluation_parameters:
            self.queue.put(params)

    monkeypatch.setattr(QuantumEvaluation, "_load_queues", fake_load_queues)
    # Patch NetworkParameters.quantum_params to return our dummy parameters.
    monkeypatch.setattr(NetworkParameters, "quantum_params",
                        lambda: mock_evaluation_parameters)
    kwargs = {"dataset_name": "Dummy", "with_noise": False}
    kwargs.update(dummy_dataset_objects)
    # Instantiate without a load_dataset keyword.
    evaluator = QuantumEvaluation(dataset_name="Dummy",
                                  **dummy_dataset_objects)
    # Clear any tasks loaded during __init__ (if any) to avoid duplication.
    evaluator.queue = Queue()
    evaluator._load_queues()
    return evaluator


def test_load_queues_when_files_missing(monkeypatch, dummy_dataset_objects,
                                        mock_evaluation_parameters):
    monkeypatch.setattr(NetworkParameters, "quantum_params",
                        lambda: mock_evaluation_parameters)
    # Instantiate without load_dataset.
    evaluator = QuantumEvaluation(dataset_name="Dummy",
                                  **dummy_dataset_objects,
                                  skip_load_queues=True)
    # Clear queue first.
    evaluator.queue = Queue()
    # Force open() to throw FileNotFoundError so that _load_queues uses
    # our evaluation parameters.
    monkeypatch.setattr("builtins.open",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            FileNotFoundError))
    evaluator._load_queues()
    assert evaluator.queue.qsize() == len(mock_evaluation_parameters), \
        (f"Expected queue size {len(mock_evaluation_parameters)}, "
         f"got {evaluator.queue.qsize()}.")


def test_store_queues(monkeypatch, dummy_dataset_objects,
                      mock_evaluation_parameters):
    evaluator = QuantumEvaluation(dataset_name="Dummy",
                                  **dummy_dataset_objects)
    evaluator.queue.put({"dummy": True})
    evaluator.queue_noise.put({"dummy_noise": True})
    evaluator.failed_tasks.put({"failed": True})
    dumps = []

    def fake_dump(data, f):
        dumps.append(data)

    monkeypatch.setattr(json, "dump", fake_dump)

    # Patch open() with a dummy context manager.
    class DummyFile:
        def __enter__(self): return self

        def __exit__(self, exc_type, exc_val, tb): pass

        def write(self, s): pass

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyFile())
    evaluator._store_queues()
    # Expect json.dump to be called for each non-empty queue.
    assert len(
        dumps) >= 3, \
        "Expected json.dump to be called for each non-empty queue."


def test_eval_noise_free(monkeypatch, mock_quantum_evaluator,
                         mock_evaluation_parameters):
    dummy_result = {"model name": "dummy_model", "accuracy": 0.9}
    dummy_params = "dummy_params"
    monkeypatch.setattr(
        mock_quantum_evaluator, "_train_and_eval_noise_free",
        lambda params: (dummy_result, dummy_params))
    monkeypatch.setattr(
        mock_quantum_evaluator, "_add_task_to_queue_noise",
        lambda task, params: mock_quantum_evaluator.queue_noise.put(
            "noise_task"
        ))
    # (Assume notify is already disabled via conftest.py.)
    # Ensure the main queue has the expected number of tasks.
    assert mock_quantum_evaluator.queue.qsize() == len(
        mock_evaluation_parameters)
    mock_quantum_evaluator._eval_noise_free()
    assert mock_quantum_evaluator.queue.empty(), \
        "Main queue should be empty after noise-free evaluation."
    assert mock_quantum_evaluator.queue_noise.get() == "noise_task", \
        "Expected 'noise_task' in noise queue."


def test_eval_with_noise(monkeypatch, dummy_dataset_objects,
                         mock_evaluation_parameters):
    kwargs = {"dataset_name": "Dummy", "with_noise": True}
    kwargs.update(dummy_dataset_objects)
    evaluator = QuantumEvaluation(**kwargs)
    # Place a dummy task into queue_noise.
    task = mock_evaluation_parameters[0].copy()
    task["parameters"] = "dummy_params"
    evaluator.queue_noise.put(task)
    # Patch _test_with_noise to return a dummy result.
    dummy_result = {"model name": "dummy_model", "accuracy": 0.85}
    monkeypatch.setattr(evaluator, "_test_with_noise",
                        lambda task, params: dummy_result)
    evaluator._eval_with_noise()
    assert evaluator.queue_noise.empty(), \
        "Queue_noise should be empty after noisy evaluation."


def test_expectation_probability():
    predictions = [[0.2, 0.8], [0.6, 0.4]]
    labels = [1, 0]
    expected = [0.8, 0.6]
    result = QuantumEvaluation.expectation_probability(predictions, labels)
    assert result == expected, f"Expected {expected}, got {result}"


def test_quantum_notification_result():
    dummy_result = {
        "accuracy": 0.8,
        "parameters": [0.1, 0.2, 0.3],
        "single_check_accuracy": 0.75,
        "triple_check_accuracy": 0.65,
        "quintil_check_accuracy": 0.60,
        "predictions": [[0.3, 0.7], [0.4, 0.6]],
        "labels": [1, 0]
    }
    notif = QuantumEvaluation.quantum_notification_result(dummy_result)
    expected_keys = {"Accuracy", "Parameters", "Single Check Accuracy",
                     "Triple Check Accuracy", "Quintil Check Accuracy",
                     "Average Prediction Probability"}
    assert expected_keys.issubset(set(notif.keys()))
    avg_prob = np.mean(np.array([0.7, 0.4]))
    np.testing.assert_allclose(notif["Average Prediction Probability"],
                               avg_prob, rtol=1e-5)


def test_save_df(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "dummy.csv"
    QuantumEvaluation._save_df(df, str(file_path))
    df_loaded = pd.read_csv(str(file_path))
    pd.testing.assert_frame_equal(df, df_loaded)
    os.remove(str(file_path))
