import logging
import numpy as np
from typing import Union, Tuple, Optional

logger = logging.getLogger(__name__)


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.

    Parameters:
    n (int): The number to check.

    Returns:
    bool: True if the number is a power of two, False otherwise.
    """
    return n > 0 and (n & (n - 1)) == 0


def handle_qubit_number(number: Union[int, float],
                        power_of_two: bool = True) -> int:
    """
    Handle and validate the number of qubits.

    Parameters:
    number (Union[int, float]): The number of qubits.
    power_of_two (bool): Whether the number of qubits must be a power of two.

    Returns:
    int: The validated number of qubits.
    """
    if not isinstance(number, (int, float)):
        raise TypeError("The number of Qubits must be an integer or float.")
    elif isinstance(number, float):
        logger.warning(
            "The number of Qubits is a float and will be converted to an "
            "integer.")
        number = int(number)

    if number < 0:
        raise ValueError("The number of Qubits must be positive.")

    if power_of_two and not is_power_of_two(number):
        raise ValueError("The number of Qubits must be a power of two.")

    return number


def handle_layer_number(number: Optional[Union[int, float]]) -> Optional[int]:
    """
    Handle and validate the number of layers.

    Parameters:
    number (Optional[Union[int, float]]): The number of layers.

    Returns:
    Optional[int]: The validated number of layers.
    """
    if number is None:
        return None

    if not isinstance(number, (int, float)):
        raise TypeError("The number of Layers must be an integer or float.")
    elif isinstance(number, float):
        logger.warning(
            "The number of Layers is a float and will be converted to an "
            "integer.")
        number = int(number)

    if number < 0:
        raise ValueError("The number of Layers must be positive.")

    return number


def calc_maximum_layer_number(number_qubits: int) -> int:
    """
    Calculate the maximum number of layers based on the number of qubits.

    Parameters:
    number_qubits (int): The number of qubits.

    Returns:
    int: The maximum number of layers.
    """
    if number_qubits <= 0:
        raise ValueError(
            "The number of qubits must be positive to calculate layers.")

    number_max_layers = int(np.log2(number_qubits))
    return number_max_layers


def handle_number_qubit_and_layer(
        qubit: Union[int, float],
        layer: Optional[Union[int, float]]
) -> Tuple[int, int]:
    """
    Handle and validate the number of qubits and layers.

    Parameters:
    qubit (Union[int, float]): The number of qubits.
    layer (Optional[Union[int, float]]): The number of layers.

    Returns:
    Tuple[int, int]: The validated number of qubits and layers.
    """
    qubit = handle_qubit_number(qubit)
    layer = handle_layer_number(layer)
    max_layer = calc_maximum_layer_number(qubit)

    if layer is None or layer > max_layer:
        layer = max_layer

    return qubit, layer


def handle_clbit_number(number: Union[int, float],
                        power_of_two: bool = True) -> int:
    """
    Handle and validate the number of classical bits (Clbits).

    Parameters:
    number (Union[int, float]): The number of Clbits.
    power_of_two (bool): Whether the number of Clbits must be a power of two.
    Default is True.

    Returns:
    int: The validated number of Clbits.

    Raises:
    ValueError: If the number is not an integer or float, or if it does not
    meet the specified conditions.
    """
    if not isinstance(number, (int, float)):
        raise ValueError("The number of Clbits must be an integer or float.")
    elif isinstance(number, float):
        logger.warning(
            "The number of Clbits is a float and will be converted to an "
            "integer."
        )
        number = int(number)

    if number < 0:
        raise ValueError("The number of Clbits must be positive.")

    if power_of_two and not is_power_of_two(number):
        raise ValueError("The number of Clbits must be a power of two.")

    return number


def handle_branch_number(number: Optional[Union[int, float]],
                         max_number_branches: Union[int, float]) -> int:
    """
    Handles the number of branches by ensuring it does not exceed the maximum
    allowed number.

    If the input number is a float, it will be converted to an integer.
    If the input number exceeds the maximum, it will be capped at the maximum.

    Parameters:
    number (Optional[Union[int, float]]): The current number of branches.
    Can be None, in which case max_number_branches is returned.
    max_number_branches (Union[int, float]): The maximum allowed number of
    branches. Can be an integer or a float, which will be converted to an
    integer.

    Returns:
    int: The processed number of branches, capped at max_number_branches if
    necessary.

    Raises:
    ValueError: If number is not an int, float, or None, or if
     max_number_branches is not an int or float.
    """
    if not isinstance(max_number_branches, (int, float)):
        raise ValueError(
            "The maximum number of branches must be an integer or float.")

    if isinstance(max_number_branches, float):
        logger.warning(
            "The maximum number of branches is a float and will be converted "
            "to an integer.")
        max_number_branches = int(max_number_branches)

    def check_max_number(num: int, max_num: int) -> int:
        if num <= max_num:
            return num
        logger.warning(
            f"The number of branches ({num}) exceeds the maximum ({max_num}). "
            f"It will be set to the maximum.")
        return max_num

    if number is None:
        return max_number_branches
    if isinstance(number, (int, float)):
        if isinstance(number, float):
            logger.warning(
                "The number of branches is a float and will be converted to "
                "an integer.")
            number = int(number)
        return check_max_number(number, max_number_branches)

    raise ValueError(
        "The number of branches must be an integer, float or None.")


def handle_qubit_group_list(
        qubit_group: list[list[int]]
) -> tuple[int, list[list[int]]]:
    """
    Handles the input qubit group for splitting and parallelizing quantum
    circuits. It checks for input types and validity of qubits.

    Parameters:
    qubit_group (list[list[int]]): A list of the qubit groups.

    Returns:
    tuple[int, list[list[int]]]:
    The number of qubits and the handled list of qubit groups.

    Raises:
    ValueError: If the list of qubit groups is not a list of lists or if the
    elements aren't integers.
    """
    if not isinstance(qubit_group, list) or not all(
            isinstance(el, list) for el in qubit_group):
        raise ValueError("The qubit groups must be a list of lists.")

    if not all([len(el) == len(qubit_group[0]) for el in qubit_group]):
        raise ValueError("The qubit groups must be the same length for all"
                         " groups")

    flattened_list = [item for sublist in qubit_group for item in sublist]

    if not all(isinstance(el, int) for el in flattened_list):
        raise ValueError(
            "The qubit group list must contain only lists of integers.")

    number_of_qubits = len(flattened_list)

    if set(flattened_list) != set(range(number_of_qubits)):
        raise ValueError(
            f"The qubit group list must contain all numbers from 0 to "
            f"{number_of_qubits - 1}.")

    return number_of_qubits, qubit_group
