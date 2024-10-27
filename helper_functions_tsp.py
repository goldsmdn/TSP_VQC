import numpy as np
import math
import copy
import graycode

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
        The total distance for the cycle represented by that bit string
    """
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
        A list of integers showing a cycle
    """
    
    bit_string_copy = copy.deepcopy(bit_string)
    end_cycle_list = []
    start_cycle_list = [i for i in range(locs)]
    end_cycle_list.append(start_cycle_list.pop(0)) #end point of cycle is always 0
    for i in range(locs-1, 1, -1):
        bin_len = find_bin_length(i)
        bin_string = []
        for count in range(bin_len):
            j = bin_len - 1 - count
            bin_string.append(bit_string_copy.pop(j))
        position = convert_binary_list_to_integer(bin_string, reverse=True, gray=gray)
        index = position % i    
        end_cycle_list.append(start_cycle_list.pop(index))
    end_cycle_list.append(start_cycle_list.pop(0)) #only one entry left if i =1
    if start_cycle_list != []:
        raise Exception('Cycle returned may not be complete')
    if bit_string_copy != []:
        raise Exception(f'bit_string not consumed {bit_string_copy} left')
    return(end_cycle_list)

#def find_stats(counts, locs, distance_array, shots, verbose=False):
#    """finds the average of the relevant counts, and the shortest distance
#    
#    Parameters
#    ----------
#    counts : dict
#        Dictionary holding the binary string and the counts observed
#    locs: integer
#        Number of locations
#    distance_array : array
#        array holding distances
#    shots: integer
#        number of shots
#    verbose: bool
#        determines in printout is made#
#
#    Returns
#    ----------
#    average : float
#        The average value
#    lowest_dist : float
#        The lowest distance
 #   """
 #   total_counts = 0
 #   total_dist = 0
#    first = True
#    for key, value in counts.items():
#        bit_list = [int(bits) for bits in key]
#        cost_fn = cost_fn_fact(locs,distance_array,verbose=False)
#        dist = cost_fn(bit_list)
#        if first == True:
#            lowest_dist = dist
#            first = False
#        else:
#            if dist < lowest_dist:
#                lowest_dist = dist
#        if verbose:
#            print(f'The distance for string {key} is {dist} and the counts are {value}')
#            print(f'The lowest_distance is {lowest_dist}')
#        total_counts += value
#        total_dist += dist * value
#    if verbose:
#        print(f'The total_counts_are {total_counts}')
#    if shots != total_counts:
#        raise Exception(f'The t {total_counts=} does not agree to the {shots=}')#
#
#    average_dist = total_dist / total_counts
#    return(average_dist, lowest_dist)

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