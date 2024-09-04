import numpy as np
from pytest import raises
#from numpy.testing import assert_array_equal
from helper_functions_tsp import validate_distance_array, find_distance
from helper_functions_tsp import convert_binary_list_to_integer
from helper_functions_tsp import check_loc_list, augment_loc_list, find_total_distance
from helper_functions_tsp import find_problem_size, convert_bit_string_to_cycle
from helper_functions_tsp import find_average
#from helper_functions_tsp import convert_bit_string_to_matrix
#from helper_functions_tsp import calculate_distance

def test_wrong_shape():
    """Checks that the correct error message is thrown for an array of the wrong shape """
    filename = 'data/wrong_shape.txt'
    locs = 5
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The distance array is not two dimensional'):
        validate_distance_array(array, locs)
    
def test_four_rows():
    """Checks that the correct error message is thrown for an array with 4 rows and 5 columns"""
    filename = 'data/four_rows.txt'
    locs = 5
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The shape of the array does not match 5 locations'):
        validate_distance_array(array, locs)

def test_six_locs():
    """Checks that the correct error message is thrown for an 5 * 5 array when there are 6 locations"""
    filename = 'data/five_d.txt'
    locs = 6
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The shape of the array does not match 6 locations'):
        validate_distance_array(array, locs)

def test_four_cols():
    """Checks that the correct error message is thrown for an array with 5 rows and 4 columns"""
    filename = 'data/four_cols.txt'
    locs = 5
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The shape of the array does not match 5 locations'):
        validate_distance_array(array, locs)

def test_unsymmetric():
    """Checks that the correct error message is thrown for an unsymmetric array"""
    filename = 'data/fri26_bad.txt'
    locs = 26
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The array is not symmetrical'):
        validate_distance_array(array, locs)

def test_distance_1():
    """Check distance read for an array element"""
    filename = 'data/four_d.txt'
    loc1 = 1
    loc2 = 2
    expected_distance = 3.5
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert expected_distance == distance

def test_distance_2():
    """Check distance read for a diagonal element"""
    filename = 'data/fri26_bad.txt'
    loc1 = 25
    loc2 = 25
    expected_distance = 0
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert expected_distance == distance

def test_distance_3():
    """Check distance read for end of row"""
    filename = 'data/four_d.txt'
    loc1 = 0
    loc2 = 3
    expected_distance = 9
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert expected_distance == distance

def test_distance_4():
    """Check distance read for end of column"""
    filename = 'data/four_d.txt'
    loc1 = 3
    loc2 = 0
    expected_distance = 9
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert distance == expected_distance 

def test_list_00():
    """Check conversion of list [0,0]"""
    bin_len = 2
    binary_list = [0,0]
    expected_result = 0
    result = convert_binary_list_to_integer(binary_list, bin_len)
    assert result == expected_result

def test_list_01():
    """Check conversion of list [0,1]"""
    bin_len = 2
    binary_list = [0,1]
    expected_result = 1
    result = convert_binary_list_to_integer(binary_list, bin_len)
    assert result == expected_result

def test_list_10():
    """Check conversion of list [1,0]"""
    bin_len = 2
    binary_list = [1,0]
    expected_result = 2
    result = convert_binary_list_to_integer(binary_list, bin_len)
    assert result == expected_result

def test_list_11():
    """Check conversion of list [1,1]"""
    bin_len = 2
    binary_list = [1,1]
    expected_result = 3
    result = convert_binary_list_to_integer(binary_list, bin_len)
    assert result == expected_result

def test_list_1110():
    """Check conversion of list [1,1,1,0]"""
    bin_len = 4
    binary_list = [1,1,1,0]
    expected_result = 14
    result = convert_binary_list_to_integer(binary_list, bin_len)
    assert result == expected_result

#def test_prepare_first_bit_string_list():
#    """Check preparation of integer list from bitstring"""     
#    bin_len = 2
#    bit_string_list = [0,0,0,1,1,0]
#    expected_result = [0,1,2]
#    result = prepare_first_list_of_locs(bit_string_list, bin_len)      
#    assert expected_result == result

def test_check_loc_list_valid1():
    """Check check location list with a valid solution"""
    locs = 4
    loc_list = [0,1,2]
    result = check_loc_list(loc_list, locs)
    expected_result = True
    assert expected_result == result

def test_check_loc_list_valid2():
    """Check check location list with a valid solution"""
    locs = 5
    loc_list = [0,1,2,3,4]
    result = check_loc_list(loc_list, locs)
    expected_result = True
    assert  expected_result == result
     
def test_check_loc_list_invalid1():
    """Check check location list with an invalid solution"""
    locs = 4
    loc_list = [0,1,1]
    result = check_loc_list(loc_list, locs)
    expected_result = False
    assert expected_result == result

def test_check_loc_list_invalid2():
    """Check check location list with an integer out of range"""
    locs = 5
    loc_list = [0, 5, 4, 7, 3]
    result = check_loc_list(loc_list, locs)
    expected_result = False
    assert expected_result == result

def test_check_loc_list_invalid3():
    """Check check location list with an integer out of range at end"""
    locs = 5
    loc_list = [0, 5, 4, 3, 7]
    result = check_loc_list(loc_list, locs)
    expected_result = False
    assert expected_result == result

def test_check_loc_list_invalid4():
    """Check check location list with an integer just out of range at end"""
    locs = 5
    loc_list = [0, 1, 4, 3, 5]
    result = check_loc_list(loc_list, locs)
    expected_result = False
    assert expected_result == result

def test_augment_loc_list1():
    """Check adding location to the end of a simple list"""
    locs = 4
    loc_list = [0,1,2]
    result = augment_loc_list(loc_list, locs)
    expected_result = [0,1,2,3]
    assert expected_result == result

def test_augment_loc_list2():
    """Check adding location to the end of a jumbled list"""
    locs = 4
    loc_list = [2,0,3]
    result = augment_loc_list(loc_list, locs)
    expected_result = [2,0,3,1]
    assert expected_result == result

def test_find_total_distance():
    """Check total distance calculation for a simple circuit"""
    filename = 'data/four_d.txt'
    distance_array = np.genfromtxt(filename)
    int_list = [0, 1, 2, 3]
    locs = 4
    expected_result = 21.0
    result = find_total_distance(int_list, locs, distance_array)
    assert expected_result == result

def test_find_problem_size_4():
    """check problem size for 4 locations"""
    locations = 4
    expected_result = (2,3)
    result = find_problem_size(locations)
    assert expected_result == result

def test_find_problem_size_5():
    """check problem size for 5 locations"""
    locations = 5
    expected_result = (2,5)
    result = find_problem_size(locations)
    assert expected_result == result

def test_find_problem_size_26():
    """check problem size for 26 locations"""
    locations = 26
    expected_result = (5,94)
    result = find_problem_size(locations)
    assert expected_result == result

def test_convert_bit_string_to_cycle_000():
    """example for 4 locations"""
    locs = 4
    bit_string = [0, 0, 0] 
    expected_result = [0, 1, 2, 3]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_001():
    """example for 4 locations"""
    locs = 4
    bit_string = [0, 0, 1] 
    expected_result = [0, 1, 3, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_010():
    """example for 4 locations"""
    locs = 4
    bit_string = [0, 1, 0] 
    expected_result = [0, 2, 1, 3]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_011():
    """example for 4 locations"""
    locs = 4
    bit_string = [0, 1, 1] 
    expected_result = [0, 2, 3, 1]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_100():
    """example for 4 locations"""
    locs = 4
    bit_string = [1, 0, 0] 
    expected_result = [0, 3, 1, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_101():
    """example for 4 locations"""
    locs = 4
    bit_string = [1, 0, 1] 
    expected_result = [0, 3, 2, 1]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_110():
    """example for 4 locations"""
    locs = 4
    bit_string = [1, 1, 0] 
    expected_result = [0, 1, 2, 3]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_111():
    """example for 4 locations"""
    locs = 4
    bit_string = [1, 1, 1] 
    expected_result = [0, 1, 3, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_3():
    """example for 5 locations"""
    locs = 5
    bit_string = [1, 1, 1, 1, 1] 
    expected_result = [0, 4, 1, 3, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

#def test_convert_string_to_matrix():
#    """Check conversion of a PT bit string to a matrix"""     
#    bit_string = [1,0,0,0,0,1,1,0,0]
#    locs = 4
#    expected_result =np.array( 
#                    [[1, 0, 0, 0],
#                     [0, 1, 0, 0],
#                     [0, 0, 0, 1],
#                     [0, 1, 0 ,0]])
#    result = convert_bit_string_to_matrix(bit_string, locs)  
#    assert_array_equal(expected_result, result)

#def test_def_calculate_penalty_sums1():
#    """Check that there are no issues if the matrix is valid"""
#    bit_string = [1,0,0,0,0,1,0,1,0]
#    locs = 4
#    expected_result = (0,0)
#    shaped_array = convert_bit_string_to_matrix(bit_string, locs)  
#    result = calculate_penalty_sums(shaped_array, locs)
#    assert expected_result == result

#def test_def_calculate_penalty_sums2():
#    """Check that an error is thrown if the matrix has two 1s in a column"""
#    bit_string = [1,0,0,0,0,1,1,0,0]
#    locs = 4
#    expected_result = (0,2)
#    shaped_array = convert_bit_string_to_matrix(bit_string, locs)  
#   result = calculate_penalty_sums(shaped_array, locs)
#    assert expected_result == result

#def test_def_calculate_penalty_sums3():
#    """Check that an error is thrown if the matrix has two 1s in a row"""
#    bit_string = [1,0,0,0,0,0,0,1,1]
#    locs = 4
#    expected_result = (2,0)
#    shaped_array = convert_bit_string_to_matrix(bit_string, locs)  
#    result = calculate_penalty_sums(shaped_array, locs)
#    assert expected_result == result

#def test_calculate_distance():
#    """Check the distance for a known cycle"""
#    filename = 'data/four_d.txt'
#    distance_array = np.genfromtxt(filename)
#    locs = 4
#    bit_string = [1,0,0,0,0,1,0,1,0]
#    shaped_array = convert_bit_string_to_matrix(bit_string, locs)  
#    expected_result = 21.0
#    result = calculate_distance(shaped_array, distance_array, locs)
#    assert expected_result == result

#def test_prepare_list_of_locs1():
#    """Check the output for a known cycle"""
#    locs = 4
#    bin_len, _= find_problem_size(locs)
#    bit_string = [1,0,1]
#    first_list_of_locs = prepare_first_list_of_locs(bit_string,bin_len)
#    result = augment_loc_list(first_list_of_locs, locs)
#    expected_result = [0,3,2,1]
#    assert expected_result == result

def test_find_average():
    counts = {'100': 145, '111': 131, '101': 183, '001': 65, '010': 84, '011': 304, '000': 59, '110': 29}
    LOCATIONS = 4
    filename = 'data/four_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    result = find_average(counts, LOCATIONS, distance_array, SHOTS)
    expected_result = 21.916
    assert expected_result == result