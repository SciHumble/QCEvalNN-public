import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from typing import Optional
from sklearn.decomposition import PCA
from qc_eval.dataset.dataset import Dataset
from qc_eval.misc.parameters import TrainingParameters
import logging

logger = logging.getLogger(__name__)


class MNIST(Dataset):
    """
    This class is to generate a dataset of two chosen numbers (via the
    parameter class) of the handwritten numbers out of the MNIST data.
    Source:
        Deng, L. (2012). The mnist database of handwritten digit images for
        machine learning research. IEEE Signal Processing Magazine, 29(6),
        141–142.
    LaTeX:
        \cite{mnist}
    """
    pixels: int = 728  # images are 28x28 in size
    name = "MNIST"

    def __init__(self,
                 feature_reduction: Optional[str] = 'resize256',
                 classes: Optional[list[int, int]] = None,
                 quantum_output: bool = False,
                 compact: bool = False,
                 **kwargs):
        """
        Args:
            feature_reduction: Is the type of encoding scheme. There are 3
            possible options ['resize256', 'pca', 'autoencoder']. If you add
            an integer as a suffix to pca or autoencoder and sets compact True
            then this is the resulting size of the X-Data/Input.
            classes: Sets the two kinds of number that gets chosen, for example
            classes = [0, 1], then the datasets only contain 0 and 1.
            quantum_output: If True the Y-Data or expected Output would be -1
            or 1, else the Y-Data/Output would be 0 or 1.
            compact: If True reduces the features, so to say compacts the
            X-Data/Input.
        """
        logger.debug(f"Init MNIST dataset with {locals()}.")
        self.feature_reduction = feature_reduction
        self.classes = classes or [1, 0]
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            tf.keras.datasets.mnist.load_data()

        # normalize
        self.x_train = self.x_train[..., np.newaxis] / 255.0
        self.x_test = self.x_test[..., np.newaxis] / 255.0

        test_filter = np.where((self.y_test == self.classes[0])
                               | (self.y_test == self.classes[1]))
        train_filter = np.where((self.y_train == self.classes[0])
                                | (self.y_train == self.classes[1]))

        self.x_train = self.x_train[train_filter]
        self.y_train = self.y_train[train_filter]
        self.x_test = self.x_test[test_filter][:TrainingParameters.test_set_size.value]
        self.y_test = self.y_test[test_filter][:TrainingParameters.test_set_size.value]

        if quantum_output:
            self.y_train = [1 if y == self.classes[0] else -1 for y in
                            self.y_train]
            self.y_test = [1 if y == self.classes[0] else -1 for y in
                           self.y_test]
        else:
            self.y_train = [1 if y == self.classes[0] else 0 for y in
                            self.y_train]
            self.y_test = [1 if y == self.classes[0] else 0 for y in
                           self.y_test]

        if feature_reduction == 'resize256':
            self.x_train = tf.image.resize(self.x_train[:], (256, 1)).numpy()
            self.x_test = tf.image.resize(self.x_test[:], (256, 1)).numpy()
            self.x_train = tf.squeeze(self.x_train).numpy()
            self.x_test = tf.squeeze(self.x_test).numpy()
        elif feature_reduction[:3].lower() == 'pca':
            self.x_train = tf.image.resize(self.x_train[:],
                                           (self.pixels, 1)).numpy()
            self.x_train = tf.squeeze(self.x_train)
            self.x_test = tf.image.resize(self.x_test[:],
                                          (self.pixels, 1)).numpy()
            self.x_test = tf.squeeze(self.x_test)
            pca_number = int(feature_reduction[3:])
            pca = PCA(pca_number)
            self.x_train = pca.fit_transform(self.x_train)
            self.x_test = pca.fit_transform(self.x_test)
            if compact:
                self._compacting()
        elif "autoencoder" in feature_reduction.lower():
            from qc_eval.dataset.autoencoder import Autoencoder
            latent_dim = int(feature_reduction[11:])
            autoencoder = Autoencoder(latent_dim)
            autoencoder.compile(optimizer='adam',
                                loss=losses.MeanSquaredError())
            autoencoder.fit(self.x_train, self.x_train,
                            epochs=10,
                            shuffle=True,
                            validation_data=(self.x_test, self.x_test))

            self.x_train = autoencoder.encoder(self.x_train).numpy()
            self.x_test = autoencoder.encoder(self.x_test).numpy()

            if compact:
                self._compacting()
        self.additional_attributes = kwargs

    def _compacting(self) -> None:
        """
        Spreads out the x values in the number realm between 0 and pi.
        Returns:
            None
        """
        self.x_train = (self.x_train - self.x_train.min()) * (
                np.pi / (self.x_train.max() - self.x_train.min()))
        self.x_test = (self.x_test - self.x_test.min()) * (
                np.pi / (self.x_test.max() - self.x_test.min()))
