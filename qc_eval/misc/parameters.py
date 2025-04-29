from enum import Enum
from pathlib import Path


class TrainingParameters(Enum):
    batch_size = 25
    learning_rate = 0.01
    steps = 200
    momentum = 0.9
    betas = (0.9, 0.999)
    save_rate = 20  # after which number of iteration an automatic save of the
    # trainings process should happen.
    autosafe_folder = Path(__file__).parent.parent / "data" / "autosafes"
    test_set_size = -1


class EmbeddingType(Enum):
    angle = "angle"
    amplitude = "amplitude"
    angle_compact = "angle_compact"


class QuantumOptimizer(Enum):
    """
    https://pennylane.ai/blog/2022/06/how-to-choose-your-optimizer/
    """
    nesterov = "nesterov"
    adam = "adam"
