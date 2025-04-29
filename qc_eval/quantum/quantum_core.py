from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import pennylane as qml
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QuantumCore(ABC):
    _qubits: list[int]
    _parameters: list[float]
    _gates: Optional[dict] = None
    _num_parameters: int
    num_qubits: int
    name: str

    @abstractmethod
    def circuit(self) -> None:
        pass

    def plot(self) -> plt.Figure:
        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit_qnode():
            self.circuit()
            return None

        fig, ax = qml.draw_mpl(circuit_qnode)()
        return fig

    @staticmethod
    def _check_length(value: list, expected_length, property_name: str):
        if len(value) < expected_length:
            raise ValueError(
                f"The length of {property_name} ({len(value)}) is shorter than"
                f" the expected length of {expected_length}."
            )
        elif len(value) > expected_length:
            original_length = len(value)
            value = value[:expected_length]
            logger.warning(
                f"The length of {property_name} ({original_length}) "
                f"is longer than the expected length of {expected_length}. "
                f"The parameters get cut to the expected length."
            )
        return value

    @property
    def parameters(self) -> list[float]:
        return self._parameters

    @parameters.setter
    def parameters(self, value: list[float]):
        logger.debug(f"Start setting parameter of {self.name}.")
        self._parameters = self._check_length(
            value,
            self.num_parameters,
            "parameters"
        )
        logger.debug(
            f"The parameters for {self.name} are set to {self.parameters}."
        )

    @property
    def qubits(self) -> list[int]:
        return self._qubits

    @qubits.setter
    def qubits(self, value: list[int]):
        logger.debug(f"Start setting qubits for {self.name}.")
        self._qubits = self._check_length(
            value,
            self.num_qubits,
            "qubits"
        )
        logger.debug(
            f"The qubits for {self.name} are set to {self.qubits}."
        )

    @property
    def gates(self) -> dict:
        pass

    @gates.getter
    def gates(self) -> dict:
        logger.debug("Getting gates")
        if self._gates is None:
            logger.debug(
                "No gates are set. Gates get pulled out of QuantumTape."
            )

            with qml.tape.QuantumTape() as tape:
                self.circuit()

            gates_list = [op.name for op in tape.operations]
            self._gates = {gate: sum(1 if gate == el else 0
                                     for el in gates_list)
                           for gate in set(gates_list)}
        return self._gates

    @property
    def num_gates(self) -> int:
        pass

    @num_gates.getter
    def num_gates(self) -> int:
        return sum(self.gates.values())

    @property
    def num_parameters(self) -> int:
        pass

    @num_parameters.getter
    def num_parameters(self) -> int:
        return self._num_parameters
