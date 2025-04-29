from enum import Enum
from qc_eval.misc.parameters import EmbeddingType
import logging

logger = logging.getLogger(__name__)


class ParamEnum(Enum):
    # general
    number_repetition = "number_of_repetition"

    # classical
    number_of_inputs = "number_of_inputs"
    # dataset -> has to be set externally
    number_of_layers = "number_of_layers"
    optimizer = "optimizer"  # adam or nesterov
    number_of_features = "number_of_features"

    # quantum
    number_of_qubits = "number_of_qubits"
    # dataset -> has to be set externally
    convolutional_circuit_number = "convolutional_circuit_number"
    pooling_circuit_number = "pooling_circuit_number"
    embedding_type = "embedding_type"
    cost_function = "cost_function"
    noise = "noise"


class NetworkParameters:
    # general
    number_of_repetition = 5

    # classical
    input_numbers = [4, 8, 16, 32]
    layer_numbers = list(range(1, 4))
    optimizer = ["adam", "nesterov"]
    feature_numbers = list(range(1, 4))

    # quantum
    qubit_numbers = [4, 8, 16]
    # when using qubit above 20 it will cause MemoryError
    convolution_layers = [1, 2, 3, 4, 5, None]
    # Done:
    pooling_layers = [1, 2, 3, None]
    embeddings = [
        # EmbeddingType.amplitude.value,
        # There is an error with the cost function with amplitude embedding
        EmbeddingType.angle.value,
        EmbeddingType.angle_compact.value
    ]
    cost_functions = ["cross entropy"]
    cnot_error_rate = []  # 0.95]  # , 0.98, 0.99]
    single_gate_error_rate = []  # 0.99]  # , 0.99, 0.999]

    # this dict is for the testing of the noise-free trained qcnn with noise
    noise_testing = {
        "controlled": [0.015, 0.01, 0.005],
        "single": [0.007, 0.004, 0.0005]
    }

    # trainings progression
    step_size = 10
    last_epoch_at = 200
    epochs = list(range(step_size, last_epoch_at + step_size, step_size))

    @classmethod
    def classical_params(cls) -> list:
        logger.debug("Generating parameter list for classical networks.")
        params = [{ParamEnum.number_of_inputs.value: input_num,
                   ParamEnum.number_of_layers.value: layer_num,
                   ParamEnum.optimizer.value: optimizer,
                   ParamEnum.number_of_features.value: feature_num,
                   ParamEnum.number_repetition.value: repetition}
                  for input_num in cls.input_numbers
                  for layer_num in cls.layer_numbers
                  for optimizer in cls.optimizer
                  for feature_num in cls.feature_numbers
                  for repetition in range(cls.number_of_repetition)]
        return params

    @classmethod
    def quantum_params(cls) -> list:
        logger.debug("Generating parameter list for quantum networks.")
        noise_list = [None] + [
            {"cnot": cnot, "single gate": single}
            for cnot, single in zip(
                cls.cnot_error_rate, cls.single_gate_error_rate
            )]
        params = [{ParamEnum.number_of_qubits.value: qubit_num,
                   ParamEnum.convolutional_circuit_number.value: conv_num,
                   ParamEnum.pooling_circuit_number.value: pool_num,
                   ParamEnum.embedding_type.value: embedding,
                   ParamEnum.cost_function.value: cost_fct,
                   ParamEnum.noise.value: noise,
                   ParamEnum.number_repetition.value: repetition}
                  for qubit_num in cls.qubit_numbers
                  for conv_num in cls.convolution_layers
                  for pool_num in cls.pooling_layers
                  for embedding in cls.embeddings
                  for cost_fct in cls.cost_functions
                  for noise in noise_list
                  for repetition in range(cls.number_of_repetition)]
        return params

    @classmethod
    def trainings_progression(cls) -> list:
        logger.debug("Generating epoch list for trainings progression.")
        return cls.epochs
