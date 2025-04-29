from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Dataset(ABC):
    """
    Abstract base class for dataset classes. Every dataset class should have
    the same properties, so that the implementation of different datasets can
    be done without too much trouble.
    """
    x_train: Any
    x_test: Any
    y_train: list[int]
    y_test: list[int]
    name: str
    feature_reduction: str

    @abstractmethod
    def __init__(self,
                 feature_reduction: str,
                 classes: Optional[list[int, int]] = None,
                 quantum_output: bool = False,
                 compact: bool = False,
                 **kwargs):
        pass

    def dataset(self) -> tuple[Any, Any, list[int], list[int]]:
        return self.x_train, self.x_test, self.y_train, self.y_test

    @classmethod
    def load_dataset(cls, file_location) -> 'Dataset':
        save_array = np.load(file_location, allow_pickle=True)
        instance = cls.__new__(cls)
        instance._fill_from_array(save_array)
        return instance

    @classmethod
    def restore_from_array(cls, array) -> 'Dataset':
        instance = cls.__new__(cls)
        instance._fill_from_array(array)
        return instance

    def save_dataset(self, file_location) -> None:
        """
        Stores the dataset as a .npy.
        Args:
            file_location: Should end with .npy

        Returns:
            None
        """
        array = self.convert_to_array()
        np.save(file_location, array)

    def convert_to_array(self) -> np.array:
        x_array = np.array([self.x_train, self.x_test], dtype=object)
        y_array = np.array([self.y_train, self.y_test], dtype=object)
        return np.array([x_array, y_array], dtype=object)

    def _fill_from_array(self, array):
        self.x_train = array[0][0]
        self.x_test = array[0][1]
        self.y_train = array[1][0]
        self.y_test = array[1][1]
