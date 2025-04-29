import os
from qc_eval.dataset.dataset import Dataset


# Dummy concrete implementation of Dataset for testing.
class DummyDataset(Dataset):
    def __init__(self,
                 feature_reduction: str,
                 classes=None,
                 quantum_output: bool = False,
                 compact: bool = False,
                 **kwargs):
        # Set required properties.
        self.feature_reduction = feature_reduction
        self.name = "DummyDataset"
        # For testing, use simple strings to avoid broadcasting issues.
        self.x_train = "train_data"
        self.x_test = "test_data"
        self.y_train = "train_labels"
        self.y_test = "test_labels"
        self.additional = kwargs


def test_dataset_method():
    """Test that dataset() returns the expected tuple."""
    ds = DummyDataset("dummy")
    x_train, x_test, y_train, y_test = ds.dataset()
    assert x_train == ds.x_train
    assert x_test == ds.x_test
    assert y_train == ds.y_train
    assert y_test == ds.y_test


def test_convert_to_array():
    """Test that convert_to_array() correctly nests dataset properties."""
    ds = DummyDataset("dummy")
    arr = ds.convert_to_array()
    # Since x_train, etc., are simple strings, numpy will not try to broadcast.
    # We expect a 2x2 array where:
    #   arr[0,0] == x_train, arr[0,1] == x_test
    #   arr[1,0] == y_train, arr[1,1] == y_test
    result = arr.tolist()
    expected = [["train_data", "test_data"], ["train_labels", "test_labels"]]
    assert result == expected


def test_restore_from_array():
    """Test that restore_from_array correctly fills the dataset attributes."""
    ds = DummyDataset("dummy")
    arr = ds.convert_to_array()
    ds_restored = DummyDataset.restore_from_array(arr)
    assert ds_restored.x_train == ds.x_train
    assert ds_restored.x_test == ds.x_test
    assert ds_restored.y_train == ds.y_train
    assert ds_restored.y_test == ds.y_test


def test_save_and_load_dataset(tmp_path):
    """Test that saving and loading a dataset recovers the original data."""
    ds = DummyDataset("dummy")
    file_path = tmp_path / "dummy_dataset.npy"
    ds.save_dataset(str(file_path))
    # Load the dataset from the saved file.
    ds_loaded = DummyDataset.load_dataset(str(file_path))
    assert ds_loaded.x_train == ds.x_train
    assert ds_loaded.x_test == ds.x_test
    assert ds_loaded.y_train == ds.y_train
    assert ds_loaded.y_test == ds.y_test
    # Clean up the temporary file.
    os.remove(str(file_path))
