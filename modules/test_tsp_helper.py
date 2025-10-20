import numpy as np
from pytest import raises
import math
from pathlib import Path

from classes.MyDataLogger import MyDataLogger, MySubDataLogger

from modules.helper_functions_tsp import(validate_distance_array, 
                                         find_distance, 
                                         convert_binary_list_to_integer, 
                                         check_loc_list, 
                                         augment_loc_list, 
                                         find_total_distance, 
                                         find_problem_size,
                                         convert_bit_string_to_cycle, 
                                         find_stats, 
                                         cost_fn_fact, 
                                         hot_start,
                                         hot_start_list_to_string, 
                                         convert_integer_to_binary_list,
                                         convert_binary_list_to_integer, 
                                         find_run_stats,
                                         )

from classes.LRUCacheUnhashable import LRUCacheUnhashable

from modules.config import NETWORK_DIR

def test_wrong_shape():
    """Checks that the correct error message is thrown for an array of the wrong shape """
    file = 'wrong_shape.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    locs = 5
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The distance array is not two dimensional'):
        validate_distance_array(array, locs)
    
def test_four_rows():
    """Checks that the correct error message is thrown for an array with 4 rows and 5 columns"""
    file = 'four_rows.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    locs = 5
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The shape of the array does not match 5 locations'):
        validate_distance_array(array, locs)

def test_six_locs():
    """Checks that the correct error message is thrown for an 5 * 5 array when there are 6 locations"""
    file = 'five_d.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    locs = 6
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The shape of the array does not match 6 locations'):
        validate_distance_array(array, locs)

def test_four_cols():
    """Checks that the correct error message is thrown for an array with 5 rows and 4 columns"""
    file = 'four_cols.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    locs = 5
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The shape of the array does not match 5 locations'):
        validate_distance_array(array, locs)

def test_unsymmetric():
    """Checks that the correct error message is thrown for an unsymmetric array"""
    file = 'fri26_bad.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    locs = 26
    array = np.genfromtxt(filename)
    with raises(Exception, match = 'The array is not symmetrical'):
        validate_distance_array(array, locs)

def test_distance_1():
    """Check distance read for an array element"""
    file = 'four_d.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    loc1 = 1
    loc2 = 2
    expected_distance = 3.5
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert expected_distance == distance

def test_distance_2():
    """Check distance read for a diagonal element"""
    file = 'fri26_bad.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    loc1 = 25
    loc2 = 25
    expected_distance = 0
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert expected_distance == distance

def test_distance_3():
    """Check distance read for end of row"""
    file = 'four_d.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    loc1 = 0
    loc2 = 3
    expected_distance = 9
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert expected_distance == distance

def test_distance_4():
    """Check distance read for end of column"""
    file = 'four_d.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    loc1 = 3
    loc2 = 0
    expected_distance = 9
    distance_array = np.genfromtxt(filename)
    distance = find_distance(loc1, loc2, distance_array)
    assert distance == expected_distance 

def test_list_00():
    """Check conversion of list [0,0]"""
    binary_list = [0,0]
    expected_result = 0
    result = convert_binary_list_to_integer(binary_list )
    assert result == expected_result

def test_list_00_gray():
    """Check conversion of list [0,0] with gray codes"""
    gray = True
    binary_list = [0,0]
    expected_result = 0
    result = convert_binary_list_to_integer(binary_list, gray)
    assert result == expected_result

def test_list_01():
    """Check conversion of list [0,1]"""
    binary_list = [0,1]
    expected_result = 1
    result = convert_binary_list_to_integer(binary_list)
    assert result == expected_result

def test_list_01_gray():
    """Check conversion of list [0,1] with gray codes"""
    gray = True
    binary_list = [0,1]
    expected_result = 1
    result = convert_binary_list_to_integer(binary_list, gray)
    assert result == expected_result

def test_list_10():
    """Check conversion of list [1,0]"""
    binary_list = [1,0]
    expected_result = 2
    result = convert_binary_list_to_integer(binary_list)
    assert result == expected_result

def test_list_10_gray():
    """Check conversion of list [1,0] with gray codes"""
    gray = True
    binary_list = [1,0]
    expected_result = 3
    result = convert_binary_list_to_integer(binary_list, gray)
    assert result == expected_result

def test_list_11():
    """Check conversion of list [1,1]"""
    binary_list = [1,1]
    expected_result = 3
    result = convert_binary_list_to_integer(binary_list)
    assert result == expected_result

def test_list_11_gray():
    """Check conversion of list [1,1]"""
    gray = True
    binary_list = [1,1]
    expected_result = 2
    result = convert_binary_list_to_integer(binary_list, gray)
    assert result == expected_result

def test_list_1110():
    """Check conversion of list [1,1,1,0]"""
    binary_list = [1,1,1,0]
    expected_result = 14
    result = convert_binary_list_to_integer(binary_list)
    assert result == expected_result

def test_list_1110_gray():
    """Check conversion of list [1,1,1,0] with gray codes"""
    gray = True
    binary_list = [1,1,1,0]
    expected_result = 11
    result = convert_binary_list_to_integer(binary_list, gray)
    assert result == expected_result

def test_list_1000_gray():
    """Check conversion of list [1,0,0,0] with gray codes"""
    gray = True
    binary_list = [1,0,0,0]
    expected_result = 15
    result = convert_binary_list_to_integer(binary_list, gray)
    assert result == expected_result

def test_check_loc_list_valid1():
    """Test validation of location list with a valid solution"""
    locs = 4
    loc_list = [0,1,2]
    result = check_loc_list(loc_list, locs)
    expected_result = True
    assert expected_result == result

def test_check_loc_list_valid2():
    """Test validation of location list with a valid solution"""
    locs = 5
    loc_list = [0,1,2,3,4]
    result = check_loc_list(loc_list, locs)
    expected_result = True
    assert  expected_result == result
     
def test_check_loc_list_invalid1():
    """Test validation of location list with an invalid solution"""
    locs = 4
    loc_list = [0,1,1]
    result = check_loc_list(loc_list, locs)
    expected_result = False
    assert expected_result == result

def test_check_loc_list_invalid2():
    """Test validation of location list with an integer out of range"""
    locs = 5
    loc_list = [0, 5, 4, 7, 3]
    result = check_loc_list(loc_list, locs)
    expected_result = False
    assert expected_result == result

def test_check_loc_list_invalid3():
    """Test validation of location list with an integer out of range at end"""
    locs = 5
    loc_list = [0, 5, 4, 3, 7]
    result = check_loc_list(loc_list, locs)
    expected_result = False
    assert expected_result == result

def test_check_loc_list_invalid4():
    """Test validation of location list with an integer just out of range at end"""
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
    filename = 'networks/four_d.txt'
    distance_array = np.genfromtxt(filename)
    int_list = [0, 1, 2, 3]
    locs = 4
    expected_result = 21.0
    result = find_total_distance(int_list, locs, distance_array)
    assert expected_result == result

def test_find_problem_size_4():
    """Check problem size for 4 locations"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 4
    sdl.formulation = 'original'
    expected_result = 3
    result = find_problem_size(sdl)
    assert expected_result == result

def test_find_problem_size_4_new():
    """Check problem size for 4 locations"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 4
    sdl.formulation = 'new'
    expected_result = 5
    result = find_problem_size(sdl)
    assert expected_result == result

def test_find_problem_size_5_new():
    """Check problem size for 5 locations"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 5
    sdl.formulation = 'new'
    expected_result = 7
    result = find_problem_size(sdl)
    assert expected_result == result

def test_find_problem_size_26():
    """Check problem size for 26 locations"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 26
    sdl.formulation = 'original'
    expected_result = 94
    result = find_problem_size(sdl)
    assert expected_result == result

def test_convert_bit_string_to_cycle_000():
    """Example for 4 locations"""
    locs = 4
    bit_string = [0, 0, 0]
    expected_result = [0, 1, 2, 3]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_001():
    """Example for 4 locations"""
    locs = 4
    bit_string = [0, 0, 1]
    expected_result = [0, 1, 3, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_010():
    """Example for 4 locations"""
    locs = 4
    bit_string = [0, 1, 0]
    expected_result = [0, 2, 1, 3]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_011():
    """Example for 4 locations"""
    locs = 4
    bit_string = [0, 1, 1]
    expected_result = [0, 2, 3, 1]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_100():
    """Example for 4 locations"""
    locs = 4
    bit_string = [1, 0, 0]
    expected_result = [0, 3, 1, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_101():
    """Example for 4 locations"""
    locs = 4
    bit_string = [1, 0, 1]
    expected_result = [0, 3, 2, 1]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_110():
    """Example for 4 locations"""
    locs = 4
    bit_string = [1, 1, 0]
    expected_result = [0, 1, 2, 3]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_111():
    """Example for 4 locations"""
    locs = 4
    bit_string = [1, 1, 1]
    expected_result = [0, 1, 3, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_111_gray():
    """Example for 4 locations"""
    locs = 4
    gray = True
    bit_string = [1, 1, 1]
    expected_result = [0, 3, 2, 1]
    result = convert_bit_string_to_cycle(bit_string, locs, gray)
    assert expected_result == result

def test_convert_bit_string_to_cycle_3():
    """Example for 5 locations"""
    locs = 5
    bit_string = [1, 1, 1, 1, 1] 
    expected_result = [0, 4, 1, 3, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_4():
    """Example for 5 locations"""
    locs = 5
    bit_string = [1, 0, 1, 1, 1] 
    expected_result = [0, 3, 1, 4, 2]
    result = convert_bit_string_to_cycle(bit_string, locs)
    assert expected_result == result

def test_convert_bit_string_to_cycle_15_gray():
    """Example for 15 locations with Gray code"""
    locs = 15
    gray = True
    expected_result = [0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    bit_string = [1, 0, 1, 1, \
                  1 ,0 ,1, 0, \
                  1, 1, 1, 0, \
                  1, 1, 1, 1, \
                  1, 1, 0, 1, \
                  1, 1, 0, 0, \
                  1, 0, 0, \
                  1, 0, 1, \
                  1, 1, 1, \
                  1, 1, 0, \
                  1, 0, 1, 1, 1]
    
    result = convert_bit_string_to_cycle(bit_string, locs, gray)
    assert expected_result == result   

def test_convert_bit_string_to_cycle_15():
    """Example for 15 locations without Gray code"""
    locs = 15
    gray = False
    expected_result = [0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    bit_string = [1, 1, 0, 1, \
                  1 ,1 ,0, 0, \
                  1, 0, 1, 1, \
                  1, 0, 1, 0, \
                  1, 0, 0, 1, \
                  1, 0, 0, 0, \
                  1, 1, 1, \
                  1, 1, 0, \
                  1, 0, 1, \
                  1, 0, 0, \
                  1, 1, 1, 0, 1]

    result = convert_bit_string_to_cycle(bit_string, locs, gray)
    assert expected_result == result 
  
def test_convert_bit_string_to_cycle_00010__not_gray():
    """Example for 4 locations"""
    locs = 4
    gray = False
    bit_string = [0, 0, 0, 1, 0]
    expected_result = [0, 2, 1, 3]
    result = convert_bit_string_to_cycle(bit_string, locs, gray, method='new')
    assert expected_result == result

def test_find_average():
    """Test find_stats in average mode"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'100': 145, '111': 131, '101': 183, '001': 65, '010': 84, '011': 304, '000': 59, '110': 29}
    sdl.locations = 4
    sdl.gray = False
    sdl.formulation = 'original'
    filename = 'networks/four_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    cost_fn = cost_fn_fact(sdl, distance_array)
    average, _, _ = find_stats(cost_fn, counts, SHOTS,)
    expected_result = 21.916
    assert expected_result == average

def test_find_lowest():
    """Test find_stats in lowest mode"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'100': 145, '111': 131, '101': 183, '001': 65, '010': 84, '011': 304, '000': 59, '110': 29}
    sdl.locations = 4
    sdl.gray = False
    sdl.formulation = 'original'
    filename = 'networks/four_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    cost_fn = cost_fn_fact(sdl, distance_array)
    _ , lowest, _ = find_stats(cost_fn, counts, SHOTS,)
    expected_result = 21.0
    assert expected_result == lowest

def test_find_average_slice1():
    """Test average slice functionality"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'11010': 1000}
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    AVERAGE_SLICE = 0.6
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 AVERAGE_SLICE, 
                                 )
    expected_result = 21.0
    assert expected_result == average

def test_find_average_slice2():
    """Test average slice functionality - ensure no change"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'11010': 1000}
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 )
    expected_result = 21.0
    assert expected_result == average

def test_find_average_slice2b():
    """Test average slice functionality - ensure no change"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'00000': 1000}
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 )
    expected_result = 25.0
    assert expected_result == average

def test_find_average_slice3():
    """Test average slice functionality - ensure no change"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'11010': 500,
              '00000': 500}
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    AVERAGE_SLICE = 0.4
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 AVERAGE_SLICE, 
                                 )
    expected_result = 21
    assert expected_result == average

def test_find_average_slice4():
    """Test average slice functionality - ensure no change"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'11010': 500,
              '00000': 500}
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    AVERAGE_SLICE = 0.6
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 AVERAGE_SLICE, 
                                 )
    expected_result = 21.3333
    assert expected_result - average < 0.0001

def test_find_average_slice5():
    """Test average slice functionality - ensure no change  """
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'11010': 200, #Energy = 21
          '00000': 300, #Energy = 25
          '01101': 500} #Energy = 19
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    AVERAGE_SLICE = 0.8
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 AVERAGE_SLICE,
                                 )
    expected_result = 20.25
    assert expected_result == average

def test_find_average_slice6():
    """Test finding average slice"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'11010': 200, #Energy = 21
          '00000': 300, #Energy = 25
          '01101': 500} #Energy = 19
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    SHOTS = 1000
    AVERAGE_SLICE = 1
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 AVERAGE_SLICE, 
                                 )
    expected_result = 21.2
    assert expected_result == average

def test_find_average_slice7():
    """Test average slice functionality - ensure no change  """
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    counts = {'11010': 200, #Energy = 21
          '00000': 300, #Energy = 25

          '01101': 500} #Energy = 19
    sdl.locations = 5
    sdl.gray = True
    sdl.formulation = 'original'
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    SHOTS = 1000
    AVERAGE_SLICE = 0.2
    cost_fn = cost_fn_fact(sdl, distance_array)
    average , _ , _ = find_stats(cost_fn, 
                                 counts, 
                                 SHOTS, 
                                 AVERAGE_SLICE, 
                                 )
    expected_result = 19.0
    assert expected_result == average

def test_hot_start_4():
    """Hot start list with four locations"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 4
    filename = 'networks/four_d.txt'
    distance_array = np.genfromtxt(filename)
    actual_result = hot_start(sdl, distance_array)
    expected_result = [0, 1, 2, 3]
    assert expected_result == actual_result

def test_hot_start_5_list():
    """Hot start list with five locations"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 5
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    actual_result = hot_start(sdl, distance_array, )
    expected_result = [0, 3, 2, 1, 4]
    assert expected_result == actual_result

def test_hot_start_5_distance():
    """Hot start distance with five locations"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 5
    filename = 'networks/five_d.txt'
    distance_array = np.genfromtxt(filename)
    list = hot_start(sdl, distance_array,)
    actual_result = find_total_distance(list, sdl.locations, distance_array)
    expected_result = 21
    assert expected_result == actual_result

def test_hot_start_list_to_string_101():
    """Hot start list with four locations in descending order"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 4
    sdl.gray = False
    sdl.formulation = 'original'
    hot_start_list = [0, 3, 2, 1]
    actual_result = hot_start_list_to_string(sdl, hot_start_list,)
    expected_result = [1, 0, 1]
    assert expected_result == actual_result

def test_hot_start_list_to_string_101_gray():
    """Hot start list with four locations in descending order with Gray code"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 4
    sdl.gray = True
    sdl.formulation = 'original'
    hot_start_list = [0, 3, 2, 1]
    actual_result = hot_start_list_to_string(sdl, hot_start_list,)
    expected_result = [1, 1, 1]
    assert expected_result == actual_result

def test_hot_start_list_to_string_15_locs_no_gray():
    """Hot start list with fifteen locations in descending order without Gray code"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 15
    sdl.formulation ='original'
    sdl.gray = False
    hot_start_list = [0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    actual_result = hot_start_list_to_string(sdl, hot_start_list, )
    expected_result = [1, 1, 0, 1, \
                       1 ,1 ,0, 0, \
                       1, 0, 1, 1, \
                       1, 0, 1, 0, \
                       1, 0, 0, 1, \
                       1, 0, 0, 0, \
                       1, 1, 1, \
                       1, 1, 0, \
                       1, 0, 1, \
                       1, 0, 0, \
                       1, 1, 1, 0, 1]
    assert expected_result == actual_result

def test_hot_start_list_to_string_15_locs_gray():
    """Hot start list with fifteen locations in descending order with Gray code"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 15
    sdl.formulation ='original'
    sdl.gray = True
    hot_start_list = [0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    actual_result = hot_start_list_to_string(sdl, hot_start_list, )
    expected_result = [1, 0, 1, 1, \
                       1 ,0 ,1, 0, \
                       1, 1, 1, 0, \
                       1, 1, 1, 1, \
                       1, 1, 0, 1, \
                       1, 1, 0, 0, \
                       1, 0, 0, \
                       1, 0, 1, \
                       1, 1, 1, \
                       1, 1, 0, \
                       1, 0, 1, 1, 1]
    assert expected_result == actual_result

def test_bit_string_list_to_bit_string():
    """Test LRC cache conversion of list to bit string"""
    bit_string_list = [0, 1, 0, 1, 0, 1]
    expected_result = '010101'
    obj = LRUCacheUnhashable()
    actual_result = obj.list_to_bit_string(bit_string_list)
    assert expected_result == actual_result

def test_binary_string_conversion():
    """Test conversion of binary string to integer and back without gray code"""
    length = 5
    gray = False
    expected_result = [i for i in range(2**length)]
    actual_result = []
    for i in expected_result:
        binary_list = convert_integer_to_binary_list(i, length, gray)
        integer = convert_binary_list_to_integer(binary_list, gray)
        actual_result.append(integer)
    assert expected_result == actual_result

def test_binary_string_conversion_gray():
    """Test conversion of binary string to integer and back with gray code"""
    length = 5
    gray = True
    expected_result = [i for i in range(2**length)]
    actual_result = []
    for i in expected_result:
        binary_list = convert_integer_to_binary_list(i, length, gray)
        integer = convert_binary_list_to_integer(binary_list, gray)
        actual_result.append(integer)
    assert expected_result == actual_result

def test_bit_string_cycle_conversion_orig():
    """Test conversion of integers to binary lists without Gray code"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 4
    sdl.formulation = 'new'
    sdl.gray = False
    f = math.factorial(sdl.locations)
    dim = find_problem_size(sdl)
    expected_result = []
    actual_result = []
    for i in range(f):
        binary_list = convert_integer_to_binary_list(i, dim, gray=sdl.gray)
        expected_result.append(binary_list)

    for binary_list in expected_result:
        cycle = convert_bit_string_to_cycle(binary_list, sdl.locations, gray=sdl.gray, method=sdl.formulation)
        new_binary_list = hot_start_list_to_string(sdl, cycle,)
        actual_result.append(new_binary_list)
    assert expected_result == actual_result

def test_bit_string_cycle_conversion_orig2():
    """Test conversion of integers to binary lists with Gray code"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 4
    sdl.formulation = 'new'
    sdl.gray = True
    f = math.factorial(sdl.locations)
    dim = find_problem_size(sdl)
    expected_result = []
    actual_result = []
    for i in range(f):
        binary_list = convert_integer_to_binary_list(i, dim, gray=sdl.gray)
        expected_result.append(binary_list)
    for binary_list in expected_result:
        cycle = convert_bit_string_to_cycle(binary_list, sdl.locations, gray=sdl.gray, method=sdl.formulation)
        new_binary_list = hot_start_list_to_string(sdl, cycle, )
        actual_result.append(new_binary_list)
    assert expected_result == actual_result


def test_lowest_list1():
    """Test run stats with two low items"""
    test_list = [100, 90, 80, 80]
    expected_result = (80, 2)
    actual_result = find_run_stats(test_list)
    assert expected_result == actual_result

def test_lowest_list2():
    """Test run stats with for identical items"""
    test_list = [100, 100, 100, 100]
    expected_result = (100, 0)
    actual_result = find_run_stats(test_list)
    assert expected_result == actual_result 