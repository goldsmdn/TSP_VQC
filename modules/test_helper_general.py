# test helper_function_general.py

from modules.helper_functions_general import (validate_list_for_duplicates,
                                              convert_list_to_dictionary,
                                              find_valid_device_loop,
                                              find_logical_to_physical_dictionary,
                                              convert_physical_to_logical_bit_string,
                                              find_qubits_measured,
                                              )   

def test_validate_list_for_duplicates():
    """Tests the validate_list_for_duplicates function"""
    input_list = [0, 1, 2, 3, 4]
    assert validate_list_for_duplicates(input_list) == True
    input_list = [0, 1, 2, 3, 4, 1]
    assert validate_list_for_duplicates(input_list) == False    

def test_convert_list_to_dictionary():
    """Tests the convert_list_to_dictionary function"""
    input_list = [0, 1, 2, 4, 3]
    expected_output = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3}
    actual_output = convert_list_to_dictionary(input_list)
    assert actual_output == expected_output

def test_find_valid_device_loop1():
    """Check that the list is read correctly"""
    qubits = 8
    actual_output = find_valid_device_loop(qubits)
    expected_output = [0, 1, 2, 3, 10, 9, 8, 7,]
    assert actual_output == expected_output

def test_find_valid_device_loop2():
    """Check that the dictionary is read correctly"""
    qubits = 8
    input_list = find_valid_device_loop(qubits)
    actual_output = convert_list_to_dictionary(input_list)
    expected_output = {0: 0, 1: 1, 2: 2, 3: 3, 4: 10, 5: 9, 6: 8, 7: 7,}
    assert actual_output == expected_output

def test_find_logical_to_physical_dictionary1():
    """Test building the look up from logical to physical qubits"""
    qubits = 8
    actual_output = find_logical_to_physical_dictionary(qubits)
    expected_output = {0: 0, 1: 1, 2: 2, 3: 3, 4: 10, 5: 9, 6: 8, 7: 7,}
    assert actual_output == expected_output

def test_find_logical_to_physical_dictionary2():
    """Test building the look up from logical to physical qubits"""
    qubits = 3
    actual_output = find_logical_to_physical_dictionary(qubits)
    expected_output = {0: 0, 1: 1, 2: 8, 3: 7,}
    assert actual_output == expected_output

def test_convert_physical_to_logical_bit_string1():
    """ Test decoding of 8 logical qubits"""
    qubits = 8
    bit_string = [0, 1, 0, 1, 1, 0, 1, 1,]
    actual_output = convert_physical_to_logical_bit_string(bit_string, qubits)
    expected_output = [0, 1, 0, 1, 1, 1, 0, 1,]
    assert actual_output == expected_output

def test_convert_physical_to_logical_bit_string2():
    """ Test decoding of three logical qubits"""
    qubits = 3
    bit_string = [0, 1, 0, 1] # note last qubit value will be returned from device, but should not be used in calculations.
    actual_output = convert_physical_to_logical_bit_string(bit_string, qubits)
    expected_output = [0, 1, 0]
    assert actual_output == expected_output 

def test_convert_physical_to_logical_bit_string3():
    """ Test decoding of three logical qubits"""
    qubits = 3
    bit_string = [0, 0, 0, 0] # note last qubit value will be returned from device, but should not be used in calculations.
    actual_output = convert_physical_to_logical_bit_string(bit_string, qubits)
    expected_output = [0, 0, 0]
    assert actual_output == expected_output 

def test_find_qubits_measured1():
    """find qubits measured for three logical qubits"""
    qubits = 3
    actual_output = find_qubits_measured(qubits)
    expected_output = 4
    assert actual_output == expected_output 

def test_find_qubits_measured2():
    """find qubits measured for three logical qubits"""
    qubits = 8
    actual_output = find_qubits_measured(qubits)
    expected_output = 8
    assert actual_output == expected_output 