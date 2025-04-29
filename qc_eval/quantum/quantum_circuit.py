from abc import ABC
from typing import Optional
import logging
from enum import Enum
import pennylane as qml
from qc_eval.quantum.quantum_core import QuantumCore

logger = logging.getLogger(__name__)


class QuantumCircuit(QuantumCore, ABC):
    _num_parameters: int = 3
    num_qubits: int = 2

    class ErrorRateEnum(Enum):
        single_gate = "single gate"
        cnot = "cnot"

    def __init__(self, qubits: list[int],
                 parameters: Optional[list[float]] = None,
                 noise: Optional[dict] = None):
        """
        Args:
            qubits: List of qubits
            parameters: List of parameters
            noise: Optional dictionary of noise
        """
        logger.debug(f"Initialize {self.name} with {locals()}.")
        if parameters is None:
            logger.debug("The parameters will be set to be a list of zeros.")
            parameters = [0] * self.num_parameters
        self.qubits = qubits
        self.parameters = parameters
        if noise is None:
            self.error_rate = None
        else:
            self.error_rate = {
                self.ErrorRateEnum.single_gate.value: noise.get("single gate",
                                                                0.0),
                self.ErrorRateEnum.cnot.value: noise.get("cnot", 0.0)
            }

    def _add_cnot_noise(self, wires: list):
        if isinstance(self.error_rate, dict):
            for wire in wires:
                qml.DepolarizingChannel(
                    self.error_rate[self.ErrorRateEnum.cnot.value], wire
                )

    def _add_single_gate_noise(self, wire: int):
        if isinstance(self.error_rate, dict):
            qml.DepolarizingChannel(
                self.error_rate[self.ErrorRateEnum.single_gate.value], wire
            )

    def _circuit_logger_message(self):
        logger.debug(f"Circuit of {self.name} get called with parameters: "
                     f"{self.parameters}.")
