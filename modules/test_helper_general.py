# test helper_function_general.py

from modules.helper_functions_general import (validate_list_for_duplicates,
                                              convert_list_to_dictionary)   

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