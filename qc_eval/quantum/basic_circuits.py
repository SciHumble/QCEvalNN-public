import logging
import pennylane as qml
import pennylane.numpy as np
from qc_eval.quantum.quantum_circuit import QuantumCircuit
from typing_extensions import Annotated
from pennylane.ops.op_math import Controlled

logger = logging.getLogger(__name__)


"""
Convention for Pooling Circuits
    Sink: qubit 1
    Source: qubit 0
"""


class BasicCircuits:
    @classmethod
    def get_circuit_dict(cls) -> dict:
        circuit_dict = {"conv": {1: cls.Conv1,
                                 2: cls.Conv2,
                                 3: cls.Conv3,
                                 4: cls.Conv4,
                                 5: cls.Conv5,
                                 None: cls.ConvDefault},
                        "pool": {1: cls.Pool1,
                                 2: cls.Pool2,
                                 3: cls.Pool3,
                                 None: cls.PoolDefault}}
        logger.debug(f"Returning PennylaneCircuits dict.")
        return circuit_dict

    @staticmethod
    def unitary_matrix(params: Annotated[list[float], 3]) -> np.ndarray:
        """
        This is the same as the UGate in qiskit.
        Source:
            https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.UGate
        Args:
            params: [theta, phi, lambda]
        """
        logger.debug(f"Unitary matrix got called with parameters: {params}.")
        return np.array([
            [np.cos(params[0] / 2),
             -np.exp(1j * params[2]) * np.sin(params[0] / 2)],
            [np.exp(1j * params[1]) * np.sin(params[0] / 2),
             np.exp(1j * (params[1] + params[2])) * np.cos(params[0] / 2)]
        ], dtype=complex)

    class CU(QuantumCircuit):
        _num_parameters = 3
        name = "controlled unitary gate"

        def circuit(self) -> None:
            """
                    qubits[0]: ─╭U(params)─┤
                    qubits[1]: ─╰●─────────┤
                    """
            self._circuit_logger_message()
            base = qml.U3(self.parameters[0],
                          self.parameters[1],
                          self.parameters[2],
                          self.qubits[0])
            Controlled(base,
                       control_wires=self.qubits[1])

    class Pool1(QuantumCircuit):
        _num_parameters = 2
        name = "pooling circuit 1"

        def circuit(self) -> None:
            """
            qubits[0]: ─╭RZ(params[0])────╭RX(params[1])─┤
            qubits[1]: ─╰●──────────────X─╰●─────────────┤

            Source:
                [Shi2024]
            """
            self._circuit_logger_message()

            qml.CRZ(self.parameters[0], wires=self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            qml.PauliX(wires=self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.CRX(self.parameters[1], wires=self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

    class Pool2(QuantumCircuit):
        _num_parameters = 6
        name = "pooling circuit 2"

        def circuit(self) -> None:
            """
            qubits[0]: ─╭U(params[:3])──X─╭U(params[3:])─┤
            qubits[1]: ─╰●────────────────╰●─────────────┤
            
            Source:
                [Hur2022]
            """
            self._circuit_logger_message()

            BasicCircuits.CU(
                self.qubits[::-1], self.parameters[:3]
            ).circuit()
            self._add_cnot_noise(self.qubits)

            qml.X(self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            BasicCircuits.CU(
                self.qubits[::-1], self.parameters[3:]
            ).circuit()
            self._add_cnot_noise(self.qubits)

    class Pool3(QuantumCircuit):
        _num_parameters = 6
        name = "pooling circuit 3"

        def u(self, params, qubit):
            qml.RZ(params[0], qubit)
            self._add_single_gate_noise(qubit)

            qml.RY(params[1], qubit)
            self._add_single_gate_noise(qubit)

            qml.RZ(params[2], qubit)
            self._add_single_gate_noise(qubit)

        def u_hermitian(self, params, qubit):
            qml.RZ(-params[2], qubit)
            self._add_single_gate_noise(qubit)

            qml.RY(-params[1], qubit)
            self._add_single_gate_noise(qubit)

            qml.RZ(-params[0], qubit)
            self._add_single_gate_noise(qubit)

        def circuit(self) -> None:
            """
            U -> ──RZ(params[0])──RY(params[1])──RZ(params[2])─

            qubits[0]:  ──U(params[0:3])───╭X──U_hermitian(params[0:3])──
            qubits[1]:  ──U(params[3:6])───╰●────────────────────────────

            Source:
                [Zheng2023]
            """
            self._circuit_logger_message()

            self.u(self.parameters[0:3], self.qubits[0])

            self.u(self.parameters[3:6], self.qubits[1])

            qml.CNOT(self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            self.u_hermitian(self.parameters[0:3], self.qubits[0])

    class PoolDefault(QuantumCircuit):
        _num_parameters = 3
        name = "default pooling circuit"

        def circuit(self) -> None:
            """
            qubits[0]: ──RZ(-1.57)─╭●──RY(params[1])─╭X──RY(params[2])─┤
            qubits[1]: ────────────╰X──RZ(params[0])─╰●────────────────┤

            Source:
                [Qiskit_QCNN]
            """
            self._circuit_logger_message()

            qml.RZ(-np.pi / 2, self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.CNOT(
                self.qubits)  # control is qubits[0] and target is qubit[1]
            self._add_cnot_noise(self.qubits)

            qml.RZ(self.parameters[0], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.RY(self.parameters[1], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.CNOT(self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            qml.RY(self.parameters[2], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

    class Conv1(QuantumCircuit):
        _num_parameters = 2
        name = "convolutional circuit 1"

        def circuit(self) -> None:
            """
            qubits[0]: ──RY(params[0])─╭●─┤
            qubits[1]: ──RY(params[1])─╰X─┤

            Source:
                [Hur2022] - Convolutional circuit 1
            """
            self._circuit_logger_message()

            qml.RY(self.parameters[0], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RY(self.parameters[1], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits)
            self._add_cnot_noise(self.qubits)

    class Conv2(QuantumCircuit):
        _num_parameters = 4
        name = "convolutional circuit 2"

        def circuit(self) -> None:
            """
            qubits[0]: ──RY(params[0])─╭X─RY(params[2])─╭●─┤
            qubits[1]: ──RY(params[1])─╰●─RY(params[3])─╰X─┤

            Source:
                [Hur2022] - Convolutional circuit 3
            """
            self._circuit_logger_message()

            qml.RY(self.parameters[0], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RY(self.parameters[1], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            qml.RY(self.parameters[2], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RY(self.parameters[3], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits)
            self._add_cnot_noise(self.qubits)

    class Conv3(QuantumCircuit):
        _num_parameters = 10
        name = "convolutional circuit 3"

        def circuit(self) -> None:
            """
            qubits[0]:  ──RX(params[0])──RZ(params[2])─╭RZ(params[4])─
            qubits[1]:  ──RX(params[1])──RZ(params[3])─╰●─────────────

                        ─╭●──────────────RX(params[6])──RZ(params[8])─┤
                        ─╰RZ(params[5])──RX(params[7])──RZ(params[9])─┤
            Source:
                [Hur2022] - Convolutional Circuit 7
            """
            self._circuit_logger_message()

            qml.RX(self.parameters[0], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RX(self.parameters[1], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.RZ(self.parameters[2], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RZ(self.parameters[3], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CRZ(self.parameters[4], self.qubits[::-1])
            self._add_single_gate_noise(self.qubits[0])
            self._add_cnot_noise(self.qubits)

            qml.CRZ(self.parameters[5], self.qubits)
            self._add_single_gate_noise(self.qubits[0])
            self._add_cnot_noise(self.qubits)

            qml.RX(self.parameters[6], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RX(self.parameters[7], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.RZ(self.parameters[8], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RZ(self.parameters[9], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

    class Conv4(QuantumCircuit):
        _num_parameters = 10
        name = "convolutional circuit 4"

        def circuit(self) -> None:
            """
            qubits[0]:  ──RX(params[0])──RZ(params[2])─╭RX(params[4])─
            qubits[1]:  ──RX(params[1])──RZ(params[3])─╰●─────────────

                        ─╭●──────────────RX(params[6])──RZ(params[8])─┤
                        ─╰RX(params[5])──RX(params[7])──RZ(params[9])─┤
            Source:
                [Hur2022] - Convolutional Circuit 8
            """
            self._circuit_logger_message()

            qml.RX(self.parameters[0], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RX(self.parameters[1], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.RZ(self.parameters[2], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RZ(self.parameters[3], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CRX(self.parameters[4], self.qubits[::-1])
            self._add_single_gate_noise(self.qubits[0])
            self._add_cnot_noise(self.qubits)

            qml.CRX(self.parameters[5], self.qubits)
            self._add_single_gate_noise(self.qubits[0])
            self._add_cnot_noise(self.qubits)

            qml.RX(self.parameters[6], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RX(self.parameters[7], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.RZ(self.parameters[8], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RZ(self.parameters[9], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

    class Conv5(QuantumCircuit):
        _num_parameters = 15
        name = "convolutional circuit 5"

        def u(self, params, qubit):
            qml.RZ(params[0], qubit)
            self._add_single_gate_noise(qubit)

            qml.RY(params[1], qubit)
            self._add_single_gate_noise(qubit)

            qml.RZ(params[2], qubit)
            self._add_single_gate_noise(qubit)

        def circuit(self) -> None:
            """
            U -> ──RZ(params[0])──RY(params[1])──RZ(params[2])─

            qubits[0]:  ──U(params[0:3])─╭X──RZ(params[6])─╭●────────────────
            qubits[1]:  ──U(params[3:6])─╰●──RY(params[7])─╰X──RY(Params[8]──

                        ──╭X──U(params[9:12])───
                        ──╰●──U(params[12:15])──
            Source:
                [Zheng2023]
            """

            self.u(self.parameters[0:3], self.qubits[0])
            self.u(self.parameters[3:6], self.qubits[1])

            qml.CNOT(self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            qml.RZ(self.parameters[6], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RY(self.parameters[7], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits)
            self._add_cnot_noise(self.qubits)

            qml.RY(self.parameters[8], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            self.u(self.parameters[9:12], self.qubits[0])
            self.u(self.parameters[12:15], self.qubits[1])

    class ConvDefault(QuantumCircuit):
        _num_parameters = 3
        name = "default convolutional circuit"

        def circuit(self) -> None:
            """
            qubits[0]: ───────────╭X──RZ(p[0])─╭●───────────╭X──RZ(1.57)─┤
            qubits[1]: ─RZ(-1.57)─╰●──RY(p[1])─╰X──RY(p[2])─╰●───────────┤

            Source:
                [Cong2019]
            """
            self._circuit_logger_message()

            qml.RZ(-np.pi / 2, self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            qml.RZ(self.parameters[0], self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])

            qml.RY(self.parameters[1], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits)
            self._add_cnot_noise(self.qubits)

            qml.RY(self.parameters[2], self.qubits[1])
            self._add_single_gate_noise(self.qubits[1])

            qml.CNOT(self.qubits[::-1])
            self._add_cnot_noise(self.qubits)

            qml.RZ(np.pi / 2, self.qubits[0])
            self._add_single_gate_noise(self.qubits[0])


if __name__ == "__main__":
    circ_dict = BasicCircuits.get_circuit_dict()
    conv = circ_dict["conv"]
    pool = circ_dict["pool"]
    for key, value in conv.items():
        print(key)
        value([0, 1]).plot().show()
    for key, value in pool.items():
        print(key)
        value([0, 1]).plot().show()
