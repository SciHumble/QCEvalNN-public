import numpy as np
import pytest
import tensorflow as tf
from qc_eval.dataset.mnist import MNIST


# Fixture to provide dummy MNIST data.
# We create a small dummy dataset with 10 training samples and 5 test samples.
@pytest.fixture
def dummy_mnist_data():
    # Dummy images: 10 training samples, 28x28, values in 0-255
    x_train = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    x_test = np.random.randint(0, 256, size=(5, 28, 28), dtype=np.uint8)
    y_test = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
    return (x_train, y_train), (x_test, y_test)


# Automatically patch tf.keras.datasets.mnist.load_data to
# return dummy_mnist_data.
@pytest.fixture(autouse=True)
def patch_mnist(monkeypatch, dummy_mnist_data):
    monkeypatch.setattr(tf.keras.datasets.mnist, "load_data",
                        lambda: dummy_mnist_data)


def test_mnist_resize256(dummy_mnist_data):
    """
    Test MNIST with 'resize256' feature reduction.
    Expects x_train to be resized to shape (num_train, 256) and labels
    to be converted to 1 (for class 1) and 0 (for class 0) when quantum_output
    is False.
    """
    dataset = MNIST("resize256", classes=[1, 0], quantum_output=False)
    # After resizing, images are resized to (num_samples, 256)
    assert dataset.x_train.shape == (10, 256)
    # Check label mapping: each label becomes 1 if equal to classes[0] (i.e. 1)
    # else 0.
    expected_y = [1 if y == 1 else 0 for y in dummy_mnist_data[0][1]]
    assert dataset.y_train == expected_y


def test_mnist_pca(dummy_mnist_data):
    """
    Test MNIST with 'pca5' feature reduction.
    Expects x_train to have shape (num_train, 5) after applying PCA with 5
    components.
    """
    dataset = MNIST("pca4", classes=[1, 0], quantum_output=False)
    assert dataset.x_train.shape == (10, 4)


def test_mnist_quantum_output(dummy_mnist_data):
    """
    Test MNIST with quantum_output=True.
    Expects labels to be mapped to 1 (if equal to classes[0]) or -1.
    """
    dataset = MNIST("resize256", classes=[1, 0], quantum_output=True)
    expected_y = [1 if y == 1 else -1 for y in dummy_mnist_data[0][1]]
    assert dataset.y_train == expected_y


def test_mnist_compacting(dummy_mnist_data):
    """
    Test the _compacting() method.
    After compacting, x_train values should be scaled between 0 and π.
    """
    dataset = MNIST("resize256", classes=[1, 0], quantum_output=False)
    dataset._compacting()
    # All values should be within [0, π].
    assert np.all(dataset.x_train >= 0)
    assert np.all(dataset.x_train <= np.pi)
    np.testing.assert_allclose(dataset.x_train.min(), 0, atol=1e-5)
    np.testing.assert_allclose(dataset.x_train.max(), np.pi, atol=1e-5)


def test_dataset_method(dummy_mnist_data):
    """
    Test that the dataset() method returns a tuple (x_train, x_test, y_train,
    y_test) matching the instance attributes.
    """
    dataset = MNIST("resize256", classes=[1, 0], quantum_output=False)
    x_train, x_test, y_train, y_test = dataset.dataset()
    np.testing.assert_array_equal(x_train, dataset.x_train)
    np.testing.assert_array_equal(x_test, dataset.x_test)
    assert y_train == dataset.y_train
    assert y_test == dataset.y_test
