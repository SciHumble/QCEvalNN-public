from abc import ABC, abstractmethod
from typing import Optional, Any
from qc_eval.quantum.quantum_core import QuantumCore


class QuantumNN(QuantumCore, ABC):
    num_layers: int
    noise: Optional[dict]
    device: Any

    @abstractmethod
    def qnode(self, x, params, embedding, cost_function):
        pass
