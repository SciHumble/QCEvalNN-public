import pytest
from qc_eval.present_data.present_data import PresentData
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


@pytest.fixture
def mock_quantum_file_location() -> Path:
    return Path(__file__).parent / "mock_quantum_data.csv"


@pytest.fixture
def mock_classical_file_location() -> Path:
    return Path(__file__).parent / "mock_classical_data.csv"


@pytest.fixture
def mock_presenter(
        mock_quantum_file_location,
        mock_classical_file_location
) -> PresentData:
    PresentData.quantum_data_location = mock_quantum_file_location
    PresentData.classical_data_location = mock_classical_file_location
    presenter = PresentData(load_quantum=True, load_classical=True)
    return presenter


def test_present_data_init(
        mock_quantum_file_location,
        mock_classical_file_location
):
    PresentData.quantum_data_location = mock_quantum_file_location
    PresentData.classical_data_location = mock_classical_file_location
    presenter = PresentData(load_quantum=True, load_classical=True)
    assert isinstance(presenter.classical_df, pd.DataFrame)
    assert isinstance(presenter.quantum_df, pd.DataFrame)
    presenter = PresentData(load_quantum=True, load_classical=False)
    assert isinstance(presenter.classical_df, pd.DataFrame)
    assert isinstance(presenter.quantum_df, pd.DataFrame)
    presenter = PresentData(load_quantum=False, load_classical=True)
    assert isinstance(presenter.classical_df, pd.DataFrame)
    assert isinstance(presenter.quantum_df, pd.DataFrame)
    presenter = PresentData(load_quantum=False, load_classical=False)
    assert isinstance(presenter.classical_df, pd.DataFrame)
    assert isinstance(presenter.quantum_df, pd.DataFrame)


def test_present_data_get_probability_accuracy(mock_presenter):
    actually = mock_presenter.get_expectation_accuracy()
    assert isinstance(actually, dict)
    actually = actually[0.0][0.0][4]
    assert actually[1]["x"] == [1.0]
    assert actually[1]["y"] == [1.0]
    assert actually[1]["z"].ndim == 2
    assert actually[1]["z"] == np.array([[0.88]])

    assert actually[3]["x"] == [1.0]
    assert actually[3]["y"] == [1.0]
    assert actually[3]["z"].ndim == 2
    assert actually[3]["z"] == np.array([[0.93]])

    assert actually[5]["x"] == [1.0]
    assert actually[5]["y"] == [1.0]
    assert actually[5]["z"].ndim == 2
    assert actually[5]["z"] == np.array([[0.95]])


def test_present_data_plot_probability_accuracy(mock_presenter):
    actually = mock_presenter.plot_expectation_accuracy(
        4, 1
    )
    assert isinstance(actually, Figure)
