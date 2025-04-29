import datetime
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from qc_eval.evaluation import Evaluation
from qc_eval.evaluation.quantum_evaluation import QuantumEvaluation
from qc_eval.training import CTraining, QTraining
from qc_eval.dataset import MNIST
from qc_eval.misc import notify
from qc_eval.evaluation.parameters_networks import *
from qc_eval.plotting import Scatter
from qc_eval import version_number


# Fixture to mock the file paths and avoid real file I/O
@pytest.fixture
def mock_paths(tmp_path):
    with patch.object(Evaluation, 'classical_file',
                      tmp_path / 'classical_results.csv'), \
            patch.object(Evaluation, 'quantum_file',
                         tmp_path / 'quantum_results.csv'), \
            patch('os.path.exists', return_value=False), \
            patch('pandas.DataFrame.to_csv') as mock_save:
        yield mock_save


# Original tests

def test_initialization(mock_paths):
    eval_instance = Evaluation(dataset="MNIST", load_dataset=False)
    assert isinstance(eval_instance.classical_df, pd.DataFrame)
    assert isinstance(eval_instance.quantum_df, pd.DataFrame)
    assert eval_instance.classical_df.empty
    assert eval_instance.quantum_df.empty


def test_init_dataset(mock_paths, monkeypatch):
    with pytest.raises(NotImplementedError):
        Evaluation(dataset="UnknownDataset", load_dataset=True)
    init_dataset = MagicMock(return_value=None)
    monkeypatch.setattr(Evaluation, "_init_dataset", init_dataset)
    Evaluation(dataset="MNIST", load_dataset=True)
    init_dataset.assert_called_once()


def test_start_evaluation(mock_paths, monkeypatch):
    # Patch _init_dataset so that Evaluation doesn't try to load real datasets.
    init_dataset = MagicMock(return_value=None)
    monkeypatch.setattr(Evaluation, "_init_dataset", init_dataset)

    classical_params = MagicMock(
        return_value=[{ParamEnum.number_of_inputs.value: 4,
                       ParamEnum.number_of_layers.value: 1,
                       ParamEnum.optimizer.value: "nesterov",
                       ParamEnum.number_of_features.value: 1,
                       ParamEnum.number_repetition.value: 1}]
    )
    monkeypatch.setattr(NetworkParameters, "classical_params",
                        classical_params)

    # Instantiate Evaluation with load_dataset=True.
    eval_instance = Evaluation(dataset="MNIST", load_dataset=True)

    # Provide dummy dataset dictionaries for all required attributes.
    dummy_data = MagicMock()  # a dummy placeholder for dataset objects
    eval_instance.pca = {4: dummy_data}
    eval_instance.pca_compact = {4: dummy_data}
    eval_instance.autoencoder = {4: dummy_data}
    eval_instance.autoencoder_compact = {4: dummy_data}
    eval_instance.resize = {4: dummy_data}

    monkeypatch.setattr(QuantumEvaluation, "__init__",
                        MagicMock(return_value=None))
    quantum_eval = MagicMock(return_value=None)
    monkeypatch.setattr(QuantumEvaluation, "evaluate", quantum_eval)

    monkeypatch.setattr(CTraining, "train", MagicMock(return_value=[0.1]))
    mock_result = {
        'accuracy': 0.9,
        'predictions': [0],
        'parameters': [0.1],
        'number of parameters': 1,
        'number of flops': 123,
        'model name': "cnn",
        'date': datetime.datetime.now()
    }
    monkeypatch.setattr(CTraining, "result",
                        MagicMock(return_value=mock_result))

    eval_instance.start(ccnn=True, qcnn=False)
    assert not eval_instance.classical_df.empty
    quantum_eval.assert_not_called()
    eval_instance.start(ccnn=False, qcnn=True)
    quantum_eval.assert_called_once()


def test_store_classical_result(mock_paths):
    eval_instance = Evaluation(dataset="MNIST", load_dataset=False)
    result = {
        "model name": "test_model",
        "accuracy": 0.90,
        "number of flops": 10000,
        "parameters": 50,
        "date": "2024-10-09"
    }
    params = {
        "number_of_inputs": 784,
        "number_of_layers": 3,
        "optimizer": "adam",
        "cost_function": "cross entropy loss",
        "dataset": MagicMock(feature_reduction="pca")
    }
    eval_instance._store_classical_result(result, trainings_time=120,
                                          params=params, loss_history=[0.8],
                                          autosafe_file="dummy_file.pt")
    assert len(eval_instance.classical_df) == 1


# Additional tests

def test_init_dataframe_existing_files(monkeypatch):
    dummy_df = pd.DataFrame({"col": [1, 2, 3]})
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    monkeypatch.setattr(pd, "read_csv", lambda path: dummy_df)
    eval_instance = Evaluation(dataset="MNIST", load_dataset=False)
    eval_instance._init_dataframe()
    assert not eval_instance.classical_df.empty
    assert not eval_instance.quantum_df.empty
    assert eval_instance.classical_df.equals(dummy_df)
    assert eval_instance.quantum_df.equals(dummy_df)


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
    notif = Evaluation.quantum_notification_result(dummy_result)
    expected_keys = {"Accuracy", "Parameters", "Single Check Accuracy",
                     "Triple Check Accuracy", "Quintil Check Accuracy",
                     "Average Prediction Probability"}
    assert expected_keys.issubset(set(notif.keys()))
    avg_prob = np.mean(np.array([0.7, 0.4]))
    np.testing.assert_allclose(notif["Average Prediction Probability"],
                               avg_prob, rtol=1e-5)


def test_save_df(tmp_path):
    eval_instance = Evaluation(dataset="MNIST", load_dataset=False)
    df = pd.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "dummy.csv"
    Evaluation._save_df(df, str(file_path))
    df_loaded = pd.read_csv(str(file_path))
    pd.testing.assert_frame_equal(df, df_loaded)
    os.remove(str(file_path))
