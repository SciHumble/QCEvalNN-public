from typing import Optional, List, Any, Tuple
from qc_eval.misc import handle_number_qubit_and_layer
from qc_eval.quantum.quantum_nn import QuantumNN
from qc_eval.quantum.basic_circuits import BasicCircuits
from qc_eval.quantum.quantum_circuit import QuantumCircuit
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
import logging
from qc_eval.misc.parameters import EmbeddingType

logger = logging.getLogger(__name__)


class QCNN(QuantumNN):
    name: str = "Quantum Convolutional Neural Network"
    num_conv_params: int
    num_pool_params: int

    def __init__(self,
                 num_qubits: int,
                 num_layers: Optional[int] = None,
                 conv_circ: Optional[int] = None,
                 pool_circ: Optional[int] = None,
                 noise: Optional[dict] = None,
                 params: Optional[list[float]] = None):
        self.num_qubits, self.num_layers = handle_number_qubit_and_layer(
            num_qubits, num_layers
        )

        self.qubits = list(range(self.num_qubits))

        self.noise = noise
        if self.noise is None:
            self.device = qml.device('default.qubit', wires=num_qubits)
        else:
            self.device = qml.device("default.mixed", wires=num_qubits)

        self.conv_circ: QuantumCircuit = \
            BasicCircuits.get_circuit_dict()["conv"][conv_circ](
                qubits=self.qubits[:2],
                noise=self.noise
            )
        self.pool_circ: QuantumCircuit = \
            BasicCircuits.get_circuit_dict()["pool"][pool_circ](
                qubits=self.qubits[:2],
                noise=self.noise
            )

        self._num_parameters = self._calc_num_parameters()
        if params is None:
            params = [0]*self.num_parameters
        self.parameters = params

        self._gates = None

    def _calc_num_parameters(self) -> int:
        """
        Calculate the total number of parameters required for the QCNN circuit.

        Returns:
            Total number of parameters across all layers.
        """
        # Each layer uses unique parameters; both convolution and pooling
        # circuits contribute parameters per layer.
        num_conv_params = self.num_layers * self.conv_circ.num_parameters
        num_pool_params = self.num_layers * self.pool_circ.num_parameters
        return num_conv_params + num_pool_params

    @staticmethod
    def _pairing_right(input_list: List[Any]) -> List[List[Any]]:
        """
        Create pairs of consecutive items from the input list.
        For example, [0, 1, 2, 3] becomes [[0, 1], [2, 3]].

        Args:
            input_list: List of items to be paired.

        Returns:
            List of pairs.
        """
        chunk_size = 2
        return [input_list[i:i + chunk_size] for i in
                range(0, len(input_list), chunk_size)]

    @staticmethod
    def _pairing_left(input_list: List[Any]) -> List[List[Any]]:
        """
        Create pairs for left pairing: the last element is paired with the
        first, and then the remaining elements are paired sequentially.
        For example, [0, 1, 2, 3] becomes [[3, 0], [1, 2]].

        Args:
            input_list: List of items to be paired.

        Returns:
            List of paired items.
        """
        chunk_size = 2
        first_pair = [input_list[-1], input_list[0]]
        remaining_pairs = [input_list[i:i + chunk_size] for i in
                           range(1, len(input_list) - 1, chunk_size)]
        return [first_pair] + remaining_pairs if remaining_pairs else [
            first_pair]

    def _convolutional_layer(self, params: List[float],
                             qubits: List[int]) -> None:
        """
        Apply the convolutional circuit to pairs of qubits using the given
        parameters.

        Args:
            params: List of parameters for the convolutional circuit.
            qubits: List of qubit indices to apply the circuit on.
        """
        self.conv_circ.parameters = params
        # Apply the circuit to each pair from right pairing.
        for qubit_pair in self._pairing_right(qubits):
            self.conv_circ.qubits = qubit_pair
            self.conv_circ.circuit()

        # Also apply the circuit to each pair from left pairing.
        for qubit_pair in self._pairing_left(qubits):
            self.conv_circ.qubits = qubit_pair
            self.conv_circ.circuit()

    def _pooling_layer(self, params: List[float], qubits: List[int]) -> None:
        """
        Apply the pooling circuit to pairs of qubits using the given
        parameters.

        Args:
            params: List of parameters for the pooling circuit.
            qubits: List of qubit indices to apply the circuit on.
        """
        self.pool_circ.parameters = params
        for qubit_pair in self._pairing_right(qubits):
            self.pool_circ.qubits = qubit_pair
            self.pool_circ.circuit()

    def _prepare_params(self) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Split the flat parameter list into separate lists for convolutional
        and pooling circuits per layer.

        Returns:
            A tuple (conv_params, pool_params) where each is a list of
            parameter lists per layer.
        """
        conv_params = []
        pool_params = []

        conv_len = self.conv_circ.num_parameters
        pool_len = self.pool_circ.num_parameters
        params_per_layer = conv_len + pool_len

        for i in range(self.num_layers):
            base = i * params_per_layer
            conv_params.append(self.parameters[base: base + conv_len])
            pool_params.append(self.parameters[base + conv_len: base + params_per_layer])

        return conv_params, pool_params

    def circuit(self):
        """
        Constructs a quantum convolutional neural network (QCNN) that reduces
        the number of qubits by half in each pooling layer.

        Parameters:
        number_of_qubits (int): The initial number of qubits in the quantum
        circuit.
        number_of_layers (Optional[int]): The number of convolutional and
        pooling layers.

        Returns:
        QuantumCircuit: The constructed QCNN as a QuantumCircuit object.
        """

        conv_parameters, pool_parameters = self._prepare_params()
        qubits = self.qubits

        for conv_params, pool_params in zip(conv_parameters, pool_parameters):
            self._convolutional_layer(conv_params, qubits)
            self._pooling_layer(pool_params, qubits)
            # After pooling, qubits are reduced (taking every second qubit).
            qubits = qubits[::2]

    def _data_embedding(self, x, embedding_type: str = "amplitude") -> None:
        """
        Embeds the input data in with amplitude or angle encoding.
        Args:
            x: Input data
            embedding_type: "angle" or "amplitude"

        Returns:
            None
        """
        # ToDo: Add other encodings, look at QCNN/embedding.py
        embedding_type = embedding_type.lower()
        if embedding_type == EmbeddingType.amplitude.value:
            AmplitudeEmbedding(x, wires=range(self.num_qubits),
                               normalize=True,
                               pad_with=0.)
        elif embedding_type == EmbeddingType.angle.value:
            AngleEmbedding(x, wires=range(self.num_qubits), rotation='Y')
        elif embedding_type == EmbeddingType.angle_compact.value:
            AngleEmbedding(x[:self.num_qubits],
                           wires=range(self.num_qubits),
                           rotation='Y')
            AngleEmbedding(x[self.num_qubits:],
                           wires=range(self.num_qubits),
                           rotation='X')
        else:
            raise NotImplementedError(
                "Currently is only the amplitude and angle encoding "
                "implemented."
            )

    def qnode(self, x, params, embedding, cost_function):
        """
        This function runs the QCNN from input to output for a given dataset
        and parameters. This will be used for training and testing the accuracy
        of the trained parameters.
        Source:
            QCNN/QCNN_circuit.py
        Args:
            x: Input
            params: Parameters of the Circuit
            embedding: "angle" or "amplitude"
            cost_function: "mse" or "cross entropy"

        Returns:
            y: predicted label
        """
        # Has to be a function inside a function, because of the decorator
        # It will be set before __init__ if it is not inside a function.
        self.parameters = params

        @qml.qnode(self.device)
        def get_qnode():
            self._data_embedding(x, embedding)
            self.circuit()
            if cost_function == 'mse':
                result = qml.expval(qml.PauliZ(0))
            elif cost_function == "cross entropy":
                result = qml.probs(wires=0)
            else:
                raise NotImplementedError(
                    'Currently are only the cost functions "mse" and '
                    '"cross entropy" implemented.'
                )
            return result

        return get_qnode()
