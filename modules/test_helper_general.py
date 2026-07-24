# test helper_function_general.py

from modules.helper_functions_general import (
    convert_binary_string_to_list,
    convert_list_to_binary_string,
    convert_list_to_dictionary,
    convert_physical_to_logical_bit_string,
    find_logical_to_physical_dictionary,
    find_physical_to_logical_dictionary,
    validate_list_for_duplicates,
)


def test_validate_list_for_duplicates():
    """Tests the validate_list_for_duplicates function"""
    input_list = [0, 1, 2, 3, 4]
    assert validate_list_for_duplicates(input_list)
    input_list = [0, 1, 2, 3, 4, 1]
    assert not validate_list_for_duplicates(input_list)


def test_convert_list_to_dictionary1():
    """Tests the convert_list_to_dictionary function"""
    input_list = [0, 1, 2, 4, 3]
    expected_output = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3}
    actual_output = convert_list_to_dictionary(input_list)
    assert actual_output == expected_output


def test_convert_list_to_dictionary2():
    """Tests the convert_list_to_dictionary function"""
    input_list = [
        24,
        25,
        34,
        33,
    ]
    expected_output = {0: 24, 1: 25, 2: 34, 3: 33}
    actual_output = convert_list_to_dictionary(input_list)
    assert actual_output == expected_output


def test_convert_list_to_dictionary_with_duplicates():
    """Tests the convert_list_to_dictionary function with duplicates"""
    input_list = [0, 1, 2, 4, 3, 1]
    try:
        convert_list_to_dictionary(input_list)
    except Exception as e:
        assert (
            str(e) == f'Qubit list {input_list} contains duplicates, not a valid input'
        )


def test_find_logical_to_physical_dictionary():
    """Tests the find_logical_to_physical_dictionary function"""
    qubits = 3
    target = 'generic_test'
    expected_dict = {0: 24, 1: 25, 2: 34, 3: 33}
    actual_dict = find_logical_to_physical_dictionary(qubits, target)
    assert actual_dict == expected_dict


def test_find_physical_to_logical_dictionary():
    """Tests the find_physical_to_logical_dictionary function"""
    qubits = 3
    target = 'generic_test'
    expected_dict = {24: 0, 25: 1, 34: 2, 33: 3}
    actual_dict = find_physical_to_logical_dictionary(qubits, target)
    assert actual_dict == expected_dict


def test_convert_physical_to_logical_bit_string1():
    """Tests the convert_physical_to_logical_bit_string function"""
    input_bitstring = [1, 1, 1, 1]
    qubits = 3
    target = 'generic_test'
    expected_output = [1, 1, 1]
    actual_output = convert_physical_to_logical_bit_string(
        input_bitstring, qubits, target
    )
    assert actual_output == expected_output


def test_convert_physical_to_logical_bit_string2():
    """Tests the convert_physical_to_logical_bit_string function"""
    input_bitstring = [1, 1, 1, 0]
    qubits = 3
    target = 'generic_test'
    expected_output = [1, 1, 0]
    actual_output = convert_physical_to_logical_bit_string(
        input_bitstring, qubits, target
    )
    assert actual_output == expected_output


def test_convert_physical_to_logical_bit_string3():
    """Tests the convert_physical_to_logical_bit_string function"""
    input_bitstring = [1, 1, 0, 1]
    qubits = 3
    target = 'generic_test'
    expected_output = [1, 1, 1]
    actual_output = convert_physical_to_logical_bit_string(
        input_bitstring, qubits, target
    )
    assert actual_output == expected_output


def test_convert_physical_to_logical_bit_string4():
    """Tests the convert_physical_to_logical_bit_string function"""
    input_bitstring = '0110'
    qubits = 3
    target = 'generic_test'
    expected_output = '010'
    actual_output = convert_physical_to_logical_bit_string(
        input_bitstring, qubits, target
    )
    assert actual_output == expected_output


def test_convert_physical_to_logical_bit_string5():
    """Tests the convert_physical_to_logical_bit_string function"""
    input_bitstring = '01010011'
    qubits = 8
    target = 'generic_test'
    expected_output = '01101100'
    actual_output = convert_physical_to_logical_bit_string(
        input_bitstring, qubits, target
    )
    assert actual_output == expected_output


def test_convert_physical_to_logical_bit_string6():
    """Tests the convert_physical_to_logical_bit_string function"""
    input_bitstring = '01011100'
    qubits = 8
    target = 'generic_test'
    expected_output = '01110010'
    actual_output = convert_physical_to_logical_bit_string(
        input_bitstring, qubits, target
    )
    assert actual_output == expected_output


def test_convert_logical_to_physical_bit_string7():
    """Tests the convert_physical_to_logical_bit_string function"""
    input_bitstring = '01000111'
    qubits = 8
    target = 'generic_test'
    expected_output = '01011100'
    actual_output = convert_physical_to_logical_bit_string(
        input_bitstring, qubits, target
    )
    assert actual_output == expected_output


def test_convert_list_to_binary_string():
    """Tests the convert_list_to_binary_string function"""
    input_list = [1, 1, 0, 1]
    expected_output = '1101'
    actual_output = convert_list_to_binary_string(input_list)
    assert actual_output == expected_output


def test_convert_binary_string_to_list():
    """Tests the binary_string_to_list function"""
    input_string = '1101'
    expected_output = [1, 1, 0, 1]
    actual_output = convert_binary_string_to_list(input_string)
    assert actual_output == expected_output
