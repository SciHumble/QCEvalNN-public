from pathlib import Path
import os
import torch
from qc_eval.evaluation.quantum_evaluation import QuantumEvaluation
from qc_eval.evaluation.parameters_networks import NetworkParameters
from qc_eval.dataset import MNIST
from qc_eval.training import QTraining
import tqdm
import logging

logger = logging.getLogger(__name__)


autosafes_path = Path(__file__).parent.parent / "data" / "autosafes"
files = os.listdir(autosafes_path)

# filtering files
filter = ["qcnn-4-2025-02-25-"]
files = [autosafes_path / file for file in files
         if (file[-3:] == ".pt"
             and any(el in file for el in filter)
             )
         ]

noises = (NetworkParameters.noise_testing["controlled"],
          NetworkParameters.noise_testing["single"])
noises = [{"cnot": c,
           "single gate": s}
          for c, s in zip(noises[0], noises[1])]

quantum_evaluator = QuantumEvaluation(
    "MNIST", dict(), dict(), dict(), dict(), dict(), skip_load_queues=True
)


def eval_with_noise(file):
    params = torch.load(file, weights_only=False)
    dataset = MNIST.restore_from_array(params["dataset"])
    params.pop("dataset")
    parameters = params["parameters"]
    dataset.feature_reduction = None
    kwargs = {
        "number_of_qubits": 4,
        "dataset": dataset,
        "convolutional_circuit_number": params["model_init_parameters"]["convolutional_circuit_number"],
        "pooling_circuit_number": params["model_init_parameters"]["pooling_circuit_number"],
        "embedding_type": params["embedding_type"],
        "cost_function": params["cost_function"],
        "parameters": parameters,
    }
    for noise in noises:
        try:
            kwargs.update(noise=noise)
            quantum_trainer = QTraining(**kwargs)
            start_time = quantum_evaluator.timestamp
            result = quantum_trainer.result()
            end_time = quantum_evaluator.timestamp
            quantum_evaluator._store_quantum_result(
                result,
                end_time-start_time,
                kwargs,
                []
            )
            return result
        except Exception as e:
            logger.info(e)
            return None


if __name__ == "__main__":
    for file in tqdm.tqdm(files):
        logger.info(eval_with_noise(file))
