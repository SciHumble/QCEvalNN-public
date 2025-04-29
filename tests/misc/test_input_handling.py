import pytest
from qc_eval.misc import handle_number_qubit_and_layer
from qc_eval.misc.input_handling import (
    is_power_of_two,
    handle_qubit_number,
    handle_layer_number,
    calc_maximum_layer_number,
    handle_clbit_number,
    handle_branch_number,
    handle_qubit_group_list
)


class TestInputHandling:
    def test_is_power_of_two(self):
        assert is_power_of_two(1) is True
        assert is_power_of_two(2) is True
        assert is_power_of_two(4) is True
        assert is_power_of_two(1024) is True
        assert is_power_of_two(0) is False
        assert is_power_of_two(3) is False
        assert is_power_of_two(10) is False

    def test_handle_qubit_number(self):
        assert handle_qubit_number(2) == 2
        assert handle_qubit_number(2.0) == 2
        with pytest.raises(TypeError):
            handle_qubit_number("two")
        with pytest.raises(ValueError):
            handle_qubit_number(-2)
        with pytest.raises(ValueError):
            handle_qubit_number(3)

    def test_handle_layer_number(self):
        assert handle_layer_number(5) == 5
        assert handle_layer_number(5.0) == 5
        assert handle_layer_number(None) is None
        with pytest.raises(TypeError):
            handle_layer_number("five")
        with pytest.raises(ValueError):
            handle_layer_number(-5)

    def test_calc_maximum_layer_number(self):
        assert calc_maximum_layer_number(2) == 1
        assert calc_maximum_layer_number(4) == 2
        assert calc_maximum_layer_number(8) == 3
        with pytest.raises(ValueError):
            calc_maximum_layer_number(0)
        with pytest.raises(ValueError):
            calc_maximum_layer_number(-4)

    def test_handle_number_qubit_and_layer(self):
        assert handle_number_qubit_and_layer(8, 2) == (8, 2)
        assert handle_number_qubit_and_layer(8, None) == (8, 3)
        assert handle_number_qubit_and_layer(8, 10) == (8, 3)
        with pytest.raises(TypeError):
            handle_number_qubit_and_layer("eight", 2)
        with pytest.raises(ValueError):
            handle_number_qubit_and_layer(8, -1)

    def test_handle_clbit_number(self):
        assert handle_clbit_number(2) == 2
        assert handle_clbit_number(4.0) == 4
        assert handle_clbit_number(8) == 8

        with pytest.raises(ValueError):
            handle_clbit_number("two")
        with pytest.raises(ValueError):
            handle_clbit_number(-2)
        with pytest.raises(ValueError):
            handle_clbit_number(3)
        with pytest.raises(ValueError):
            handle_clbit_number(5.0)

    def test_handle_clbit_number_no_power_of_two(self):
        assert handle_clbit_number(3, power_of_two=False) == 3
        assert handle_clbit_number(6.0, power_of_two=False) == 6

        with pytest.raises(ValueError):
            handle_clbit_number("three", power_of_two=False)
        with pytest.raises(ValueError):
            handle_clbit_number(-3, power_of_two=False)

    def test_handle_branch_number(self):
        assert handle_branch_number(None, 5) == 5
        assert handle_branch_number(3, 5) == 3
        assert handle_branch_number(7, 5) == 5
        assert handle_branch_number(4.5, 5) == 4
        assert handle_branch_number(7.5, 5) == 5
        assert handle_branch_number(3, 4.5) == 3
        assert handle_branch_number(7, 4.5) == 4
        assert handle_branch_number(7.5, 4.5) == 4

        try:
            handle_branch_number("a string", 5)
        except ValueError as e:
            assert str(
                e) == ("The number of branches must be an integer, float or "
                       "None.")

        try:
            handle_branch_number(3, "a string")
        except ValueError as e:
            assert str(
                e) == ("The maximum number of branches must be an integer or "
                       "float.")

    def test_handle_qubit_groups_valid_input(self):
        qubit_group = [[0, 1], [2, 3]]
        expected_output = (4, [[0, 1], [2, 3]])
        assert handle_qubit_group_list(qubit_group) == expected_output

    def test_handle_qubit_groups_invalid_type(self):
        with pytest.raises(ValueError,
                           match="The qubit groups must be a list of lists."):
            handle_qubit_group_list([1, 2, 3])

    def test_handle_qubit_groups_non_integer_elements(self):
        qubit_group = [[0, 1], [2, 'a']]
        with pytest.raises(ValueError,
                           match="The qubit group list must contain only lists of integers."):
            handle_qubit_group_list(qubit_group)

    def test_handle_qubit_groups_missing_qubits(self):
        qubit_group = [[0, 1], [3, 4]]
        with pytest.raises(ValueError,
                           match="The qubit group list must contain all numbers from 0 to 3."):
            handle_qubit_group_list(qubit_group)

    def test_handle_qubit_groups_unsorted_qubits(self):
        qubit_group = [[1, 0], [3, 2]]
        expected_output = (4, [[1, 0], [3, 2]])
        assert handle_qubit_group_list(qubit_group) == expected_output

    def test_handle_qubit_groups_unequal_in_size(self):
        qubit_group = [[1, 0], [2, 3, 4]]
        with pytest.raises(ValueError):
            handle_qubit_group_list(qubit_group)


if __name__ == "__main__":
    pytest.main()
