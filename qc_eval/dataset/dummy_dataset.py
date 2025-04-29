import numpy as np
from qc_eval.dataset.dataset import Dataset


class DummyDataset(Dataset):
    """
    This dummy dataset is for testing or when a dataset without meaning is needed
    """
    name = "Dummy"
    number_of_training_datasets = 10
    number_of_testing_datasets = 5

    def __init__(self, number_of_features: int):
        self.x_train = np.random.rand(
            self.number_of_training_datasets,
            number_of_features
        )
        self.y_train = np.random.randint(
            low=0,
            high=1+1,
            size=(self.number_of_training_datasets, )
        ).tolist()
        self.x_test = np.random.rand(
            self.number_of_testing_datasets,
            number_of_features
        )
        self.y_test = np.random.randint(
            low=0,
            high=1+1,
            size=(self.number_of_testing_datasets, )
        ).tolist()
