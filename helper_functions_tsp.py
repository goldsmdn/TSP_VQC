import numpy as np
import math
import copy
import graycode
import csv
from itertools import count

def read_index(filename, encoding):
    """Reads CSV file and returns and dictionary
     
    Parameters
    ----------
    filename : str
        The filename of the CSV file.  
    encoding : str
        The expected coding.  If this is missed 
        get odd charactors at start of the file

    Returns
    -------
    dict : dict
        A dictionary with the contents on the CSV file
    """
    dict = {}
    index = count()
    with open( filename, 'r', encoding=encoding) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dict[next(index)] = row
    return(dict)

def read_file_name(locations, data_sources):
    """Find the filename for a certain number of locations
    
    Parameters
    ----------
    locations : int
        Number of locations, or vertices
    data_sources : dict
        Dictionary listing the filename for each problem siz

    Returns
    ----------
    filename : string
        The filename for that problem size
    """
    filename = data_sources[locations]['file']
    return(filename)       

def validate_distance_array(array, locs):
    """Validates the distance array and raises an Exception if the 
    array is not valid.  Checks the array is the correct shape, and
    is symmetric
    
    Parameters
    ----------
    array : array
        Numpy symmetric array with distances
    locs : int
        Number of locations or vertices

    """
    if len(np.shape(array)) !=2:
        raise Exception('The distance array is not two dimensional')
    for index in [0,1]:
        if np.shape(array)[index] != locs:
            raise Exception(f'The shape of the array does not match {locs} locations')
    #check symmetry
    for i in range(locs):
        for j in range(locs):
            if array[i,j] != array[j,i]:
                raise Exception('The array is not symmetrical')

def find_distance(loc1, loc2, distance_array,verbose=False):
    """Finds the distance between locations using the distance matrix
    
    Parameters
    ----------
    loc1 : int
        First location
    loc2 : int
        Second location

    Returns
    ----------
    distance : Float
        The distance between two postal codes
    """
    
    distance = distance_array[loc1][loc2]
    if verbose:
        print(f'The distance from location {loc1} to location {loc2} is {distance}')
    return(distance)

def find_bin_length(i):
    """find the length of a binary string to represent integer i"""
    bin_len = math.ceil((np.log2(i)))
    return(bin_len) 

def find_problem_size(locations):
    """Finds the number of binary variables needed
    
    Parameters
    ----------
    locations : int
        Number of locations

    Returns
    ----------
    bin_len : int
        Length of the longest bit string needed to hold the next integer in the cycle
    pb_dim : int
        Length of the Bit string needed to store the problem
    """
    pb_dim = 0
    bin_len = find_bin_length(locations)  
    #ignore first and last item as these are moved by default
    for i in range(1, locations):
        bin_len = find_bin_length(i)
        pb_dim += bin_len
    return(bin_len, pb_dim)

def convert_binary_list_to_integer(binary_list, reverse=False, gray=False):
    """Converts list of binary numbers to an integer
    
    Parameters
    ----------
    binary_list : list
        List of binary numbers
    bin_len: int
        Length of the list of binary numbers
    reverse: bool
        reverse the string

    Returns
    ----------
    result : int
        The integer represented by the concatenated binary string
    """

    if reverse:
       binary_list.reverse()
    string = ''
    for item in binary_list:
        string += str(item)
    result= int(string, base=2)
    if gray:
        result = graycode.gray_code_to_tc(result)
    return(result)

def check_loc_list(loc_list, locs):
    """checks that the location list is a valid cycle with no repetition of nodes.
    
    Parameters
    ----------
    loc_list : list
        A list of locations with a length one less than the number of locations in the problem
    locs : int
        The total number of locations in the problem

    Returns
    ----------
    valid : Boolean
        Whether the loc_list is a valid cycle
    """

    valid = True
    for i in range(len(loc_list)):
        if loc_list[i] > (locs - 1):
            valid = False
    for i in range(0, locs-1):
        for j in range(0, locs-1):
            if i != j:
                if loc_list[i] == loc_list[j]:
                    valid = False
    return(valid)
    

def augment_loc_list(loc_list, locs):
    """completes the cycle by adding the missing location to the end of the cycle.
    
    Parameters
    ----------
    loc_list : list
        A list of locations with a length one less than the number of locations in the problem
    locs : int
        The total number of locations in the problem

    Returns
    ----------
    valid : loc_list
        The original location list with the missing node list added to the end of the cycle

    """
    
    full_list = [i for i in range(0,locs)]
    for item in full_list:
        if item not in loc_list:
            add_item = item
    loc_list.append(add_item)
    return(loc_list)

def find_total_distance(int_list, locs, distance_array, verbose=False):
    """finds the total distance for a valid formatted bit string representing a cycle.
    If relevant applies a distance penalty
    
    Parameters
    ----------
    int_list : list of integers
        A list representing a valid cycle
    locs: int
        The number of locations
    distance_array : array
        Numpy symmetric array with distances

    Returns
    ----------
    total_distance : float
        The total distance for the cycle represented by that integer list
    """
    if len(int_list) != locs:
        raise Exception(f'The list supplied has {len(int_list)} entries and {locs} are expected')

    total_distance = 0
    for i in range(0, locs):
        if i < locs-1:
            j = i + 1
        elif i == locs - 1:
            #complete cycle
            j = 0
        else:
            raise Exception('Unexpected values of i in loop')
        distance = find_distance(int_list[i], int_list[j], distance_array, verbose)
        total_distance += distance
    return total_distance

def cost_fn_fact(locs,distance_array,gray=False,verbose=False):
    """ returns a function"""
    def cost_fn(bit_string):
        """returns the value of the objective function for a bit_string"""
        full_list_of_locs = convert_bit_string_to_cycle(bit_string, locs, gray)
        total_distance = find_total_distance(full_list_of_locs, locs, distance_array)
        valid = check_loc_list(full_list_of_locs,locs)
        if not valid:
            raise Exception('Algorithm returned incorrect cycle')
        else:
            if verbose:
                print(f'bitstring = {bit_string}, full_list_of_locs = {full_list_of_locs}, distance = {total_distance}')
            return total_distance
    return(cost_fn)

def convert_bit_string_to_cycle(bit_string,locs,gray=False):
    """converts a bit string to a cycle.
    
    Parameters
    ----------
    bit_string : list
        A list of zeros and ones produced by the quantum computer
    locs: int
        The number of locations

    Returns
    ----------
    end_cycle_list : list
        A list of integers showing a cycle.  The bit string is processed from left to right
    """
    
    bit_string_copy = copy.deepcopy(bit_string)
    end_cycle_list = []
    start_cycle_list = [i for i in range(locs)]
    end_cycle_list.append(start_cycle_list.pop(0)) #end point of cycle is always 0
    for i in range(locs-1, 1, -1):
        bin_len = find_bin_length(i)
        bin_string = []
        #print(f'i = {i}, bin_len ={bin_len}')
        for count in range(bin_len):
            #j = bin_len - 1 - count
            #bin_string.append(bit_string_copy.pop(j))
            #position = convert_binary_list_to_integer(bin_string, reverse=True, gray=gray)
            #print(f'count = {count} old bit_string_copy = {bit_string_copy}')
            bin_string.append(bit_string_copy.pop(0)) #pop the most left hand item
            #print(f'new bit_string_copy = {bit_string_copy}')
            #print(f'bit_string = {bin_string}')
        position = convert_binary_list_to_integer(bin_string, gray=gray)
        #print(f'position = {position}')
        index = position % i    
        end_cycle_list.append(start_cycle_list.pop(index))
    end_cycle_list.append(start_cycle_list.pop(0)) #only one entry left if i =1
    if start_cycle_list != []:
        raise Exception('Cycle returned may not be complete')
    if bit_string_copy != []:
        raise Exception(f'bit_string not consumed {bit_string_copy} left')
    return(end_cycle_list)

def find_stats(cost_fn, counts, shots, verbose=False):
    """finds the average energy of the relevant counts, and the lowest energy
    
    Parameters
    ----------
    counts : dict
        Dictionary holding the binary string and the counts observed
    shots: integer
        number of shots
    verbose: bool
        determines in printout is made

    Returns
    ----------
    average : float
        The average energy
    lowest_dist : float
        The lowest energy
    """
    total_counts = 0
    total_energy = 0
    first = True
    for key, value in counts.items():
        bit_list = [int(bits) for bits in key]
        energy = cost_fn(bit_list)
        if first == True:
            lowest_energy = energy
            first = False
            lowest_energy_bit_string = bit_list
        else:
            if energy < lowest_energy:
                lowest_energy = energy
                lowest_energy_bit_string = bit_list
        if verbose:
            print(f'The energy for string {key} is {energy} and the counts are {value}')
            print(f'The lowest_distance is {energy}')
            print(f'The lowest energy bit string is {lowest_energy_bit_string }')
        total_counts += value
        total_energy += energy * value
    if verbose:
        print(f'The total_counts_are {total_counts}')
    if shots != total_counts:
        raise Exception(f'The t {total_counts=} does not agree to the {shots=}')

    average_energy = total_energy / total_counts
    return(average_energy, lowest_energy, lowest_energy_bit_string)

def hot_start(distance_array: np.array, locs: int) -> list:
    """finds a route from a distance array where the distance to the next point is the shortest available"""
    validate_distance_array(distance_array, locs)
    remaining_cycle_list = [i for i in range(locs)]
    end_cycle_list = []
    end_cycle_list.append(remaining_cycle_list.pop(0)) #start point of cycle is always 0
    #print(distance_array[first_item])
    #print(f'remaining cycle list is {remaining_cycle_list}')
    next_row = 0
    for i in range(locs-1):
        for j, column in enumerate(remaining_cycle_list):
            distance = distance_array[next_row][column]
            if j == 0:
                #print(f'i = {j} remaining_cycle_list[j] {remaining_cycle_list[j]}, distance = {distance}')
                #first time around the loop
                arg_min = j
                lowest_distance = distance
            else:
                #print(f'i = {j} remaining_cycle_list[j] {remaining_cycle_list[j]}, distance = {distance}')
                if distance < lowest_distance:
                    arg_min = j
                    #print(f'Arg min is now {arg_min}')
                    lowest_distance = distance
        next_row = remaining_cycle_list.pop(arg_min)
        #print(f'Next row is {next_row}')
        end_cycle_list.append(next_row)
    #end_cycle_list.append(0) #cycle always ends at zero
    return(end_cycle_list)

def hot_start_list_to_string(hot_start_list: list, locations: int, gray:bin) -> list:
    """invert the hot start integer list into a string"""
    if len(hot_start_list) != locations:
        raise Exception(f'The hot start list should be length {locations}')
    
    first_item = hot_start_list.pop(0)
    #remove the first item for the list which should be zero
    if first_item != 0:
        raise Exception(f'The first item of the list must be zero')
    
    initial_list = [i for i in range(1, locations)]
    print(f'initial_list {initial_list}')
    
    total_binary_string = ''
    result_list = []
    print(f'hot_start_list {hot_start_list})
    
    for i, integer in enumerate(hot_start_list):
        bin_len = find_bin_length(len(initial_list))
        if bin_len > 0:
        #find the index of integer in hot start list
            print(f'bin_len - {bin_len}')
            index = initial_list.index(integer)
            print(f'index = {index} integer = {integer}, hot_start_list = {hot_start_list}, initial_list, {initial_list}')
            if gray:
               # binary_string = bin(graycode.tc_to_gray_code(index))
               binary_string = 
            else:
                binary_string = bin(index)
            print(binary_string)
            binary_string = binary_string[2:] #remove the 0b charactor
            print(binary_string)
            binary_string = binary_string.zfill(bin_len)
            print(binary_string)
            total_binary_string += binary_string
            initial_list.pop(index)
    #result_list = list(total_binary_string)
    for i in range(len(total_binary_string)):
        result_list.append(int(total_binary_string[i]))
    #result_list = [int(str) for str in total_binary_string.split()]
    return(result_list)
    




    
