import numpy as np
import math
import copy
import graycode
import csv
from itertools import count
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2
import random

from collections import OrderedDict
from functools import wraps
from typing import Callable # Import Callable for type hinting

from modules.config import VERBOSE
from classes.LRUCacheUnhashable import LRUCacheUnhashable

def read_index(filename: str, encoding: str) -> dict:
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

def read_file_name(locations: int, data_sources: dict) -> str:
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

def validate_distance_array(array :np.array, locs: int):
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

def find_distance(loc1: int, loc2: int, distance_array: np.array, verbose: bool=False) -> float:
    """Finds the distance between locations using the distance matrix
    
    Parameters
    ----------
    loc1 : int
        First location
    loc2 : int
        Second location
    distance_array : np.array
        An array containing the distances between locations

    Returns
    ----------
    distance : Float
        The distance between two postal codes
    """
    
    distance = distance_array[loc1][loc2]
    if verbose:
        print(f'The distance from location {loc1} to location {loc2} is {distance}')
    return(distance)

def find_bin_length(i: int) -> int:
    """find the length of a binary string to represent integer i"""
    bin_len = math.ceil((np.log2(i)))
    return(bin_len) 

def find_problem_size(locations:int) -> tuple:
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
        Length of the bit string needed to store the problem
    """
    pb_dim = 0
    bin_len = find_bin_length(locations)  
    #ignore first and last item as these are moved by default
    for i in range(1, locations):
        bin_len = find_bin_length(i)
        pb_dim += bin_len
    return(bin_len, pb_dim)

def convert_binary_list_to_integer(binary_list: list, gray:bool=False)->int:
    """Converts list of binary numbers to an integer
    
    Parameters
    ----------
    binary_list : list
        List of binary numbers

    Returns
    ----------
    result : int
        The integer represented by the concatenated binary string
    """
    string = ''
    for item in binary_list:
        string += str(item)
    result= int(string, base=2)
    if gray:
        result = graycode.gray_code_to_tc(result)
    return(result)

def check_loc_list(loc_list:list, locs:int) -> bool:
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
    
def augment_loc_list(loc_list:list, locs:int)-> list:
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

def find_total_distance(int_list: list, locs: int, distance_array :np.array)-> float:
    """finds the total distance for a valid formatted bit string representing a cycle.
    
    Parameters
    ----------
    int_list : list of integers
        A list representing a valid cycle
    locs: int
        The number of locations
    distance_array : array
        Numpy symmetric array with distances between locations

    Returns
    -------
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
        distance = find_distance(int_list[i], int_list[j], distance_array)
        total_distance += distance
    return total_distance

#def list_to_bit_string(bit_string_list):
#    if type(bit_string_list) != list:
#        raise Exception(f'{bit_string_list} is not a list')
#    bit_string = ''
#    for i in range(len(bit_string_list)):
#        bit_string += str(bit_string_list[i])
#    return(bit_string)

#def bit_string_to_list(bit_string):
#    if type(bit_string) != str:
#        raise Exception(f'{bit_string} is not a string')
#    bit_string_list = [int(bit) for bit in bit_string]
 #   return(bit_string_list)

#def lru_cache_unhashable(orig_func, maxsize=10_000):
#    """
#    A decorator that caches results for functions with unhashable arguments.
#    Uses an OrderedDict to implement a simple LRU eviction policy.
#    """
#    cache = OrderedDict()
#    @wraps(orig_func)
#    def wrapper(bit_string_list):
#        """if there is a key in the cache use it, 
#        otherwise, compute and add to the cache"""
#        key = list_to_bit_string(bit_string_list)
#        if key in cache:
#            if VERBOSE:
#                print(f'Reading cache with key = {key}')
#                print(f'Cache is now {cache}')
#            # Mark this key as recently used
#            cache.move_to_end(key)
#            result =  cache[key]
#        else:
#            result = orig_func(bit_string_list)
#            cache[key] = result # Store the result in the cache
#            if VERBOSE:
#                print(f'Updating cache with key = {key}')
#            # Evict the least recently used item if the cache is too big
#            if len(cache) > maxsize:
#                cache.popitem(last=False)
#                if VERBOSE:
#                    print('popping item from cache')
#        return result
#    return wrapper

def cost_fn_fact(locs: int, 
                 distance_array: np.array, 
                 gray: bool=False, 
                 verbose: bool=False)-> Callable[[list], int]:
    """ returns a function

    Parameters
    ----------
    locs: int
        The number of locations in the problem
    distance_array: array
        Numpy symmetric array with distances between locations
    gray: bool
        If True Gray codes are used
    verbose: bool
        If True more information is printed out
    
    Returns
    -------
    cost_fn: cost function
        A function of a bit string evaluating a distance for that bit string
    
    """
    @LRUCacheUnhashable
    def cost_fn(bit_string):
        """returns the value of the objective function for a bit_string"""
        full_list_of_locs = convert_bit_string_to_cycle(bit_string, locs, gray)
        total_distance = find_total_distance(full_list_of_locs, locs, distance_array)
        valid = check_loc_list(full_list_of_locs,locs)
        if not valid:
            raise Exception('Algorithm returned incorrect cycle')
        else:
            if verbose:
                print(f'bitstring = {bit_string}, full_list_of_locs = {full_list_of_locs}, total_distance = {total_distance}')
            return total_distance
    return(cost_fn)

def convert_bit_string_to_cycle(bit_string: list, locs: int, gray: bool=False) -> list:
    """converts a bit string to a cycle.
    
    Parameters
    ----------
    bit_string : list
        A list of zeros and ones produced by the quantum computer
    locs: int
        The number of locations
    gray: bool
        If True Gray codes are used

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
        for count in range(bin_len):
            bin_string.append(bit_string_copy.pop(0)) #pop the most left hand item
        position = convert_binary_list_to_integer(bin_string, gray=gray)
        index = position % i    
        end_cycle_list.append(start_cycle_list.pop(index))
    end_cycle_list.append(start_cycle_list.pop(0))
    if start_cycle_list != []:
        raise Exception('Cycle returned may not be complete')
    if bit_string_copy != []:
        raise Exception(f'bit_string not consumed {bit_string_copy} left')
    return(end_cycle_list)

def find_stats(cost_fn, counts: dict, shots: int, 
               average_slice: float=1, verbose: bool=False)-> tuple:
    """finds the average energy of the relevant counts, and the lowest energy
    
    Parameters
    ----------
    cost_fn: function
        A function of a bit string evaluating an energy(distance) for that bit string
    counts : dict
        Dictionary holding the binary string and the counts observed
    shots: integer
        number of shots
    average_slice: float
        average over this slice of the energy.  eg
        If average_slice = 1 then average over all energies.  
        If average_slice = 0.2 then average over the bottom 20% of energies
    verbose: bool
        determines if data is printed out

    Returns
    ----------
    average : float
        The average energy
    lowest_dist : float
        The lowest energy
    lowest_energy_bit_string: list
        A list of the bits for the lowest energy bit string
    """
    if average_slice > 1:
        raise Exception(f'The average_slice must be less or equal to 1, not {average_slice}')
    elif average_slice <=0:
        raise Exception(f'The average_slice must be greater than zero, not {average_slice}')
    elif average_slice == 1:
        slicing = False
    else:
        slicing = True
    total_counts, total_energy = 0, 0
    first = True
    if slicing:
        energy_dict = {}
    for key, count in counts.items():
        #bit_list = bit_string_to_list(key)
        bit_list = [int(bits) for bits in key]
        energy = cost_fn(bit_list)
        if slicing:
            #if already in dictionary increment
            if energy in energy_dict.keys():
                energy_dict[energy] +=count
            else:
                energy_dict[energy] =count
        if first == True:
            lowest_energy = energy
            first = False
            lowest_energy_bit_string = bit_list
        else:
            if energy < lowest_energy:
                lowest_energy = energy
                lowest_energy_bit_string = bit_list
        if verbose:
            print(f'The energy for string {key} is {energy} and the counts are {count}')
            print(f'The lowest_distance is {lowest_energy}')
            print(f'The lowest energy bit string is {lowest_energy_bit_string }')
            if slicing:
                print(f'Slicing in progress and unsorted energy_dict = {energy_dict}')

        total_counts += count
        total_energy += energy * count

    if shots != total_counts:
        raise Exception(f'The total_counts {total_counts=} does not agree to the {shots=}')

    if slicing:
        accum, total_energy, total_counts = 0, 0, 0
        stop = shots * average_slice
        sorted_energy_dict = dict(sorted(energy_dict.items()))
        if verbose:
            print(f'The sorted energy dict is {sorted_energy_dict}')
        for energy, count in sorted_energy_dict.items():
            if accum < stop:
                if accum + count < stop:
                    total_energy += energy * count
                    total_counts += count
                else:
                    total_counts += stop - accum
                    total_energy += energy * (stop - accum)
            accum += count
    average_energy = total_energy / total_counts
    if verbose:
        print(f'Slicing = {slicing}. The total_counts_are {total_counts}')
        print(f'Returning average_energy, lowest_energy, lowest_energy_bit_string')
        print(f'{average_energy}, {lowest_energy}, {lowest_energy_bit_string}')

    return(average_energy, lowest_energy, lowest_energy_bit_string)

def hot_start(distance_array: np.array, locs: int) -> list:
    """finds a route from a distance array where the distance to the next point is the shortest available
    
    Parameters
    ----------
    locs: int
        The number of locations in the problem
    distance_array: array
        Numpy symmetric array with distances between locations

    Returns
    -------
    end_cycle_list: list
        A list of integers showing the an estimate of the lowest cycle
    
    """
    validate_distance_array(distance_array, locs)
    remaining_cycle_list = [i for i in range(locs)]
    end_cycle_list = []
    end_cycle_list.append(remaining_cycle_list.pop(0)) #start point of cycle is always 0
    next_row = 0
    for i in range(locs-1):
        for j, column in enumerate(remaining_cycle_list):
            distance = distance_array[next_row][column]
            if j == 0:
                arg_min = j
                lowest_distance = distance
            else:
                if distance < lowest_distance:
                    arg_min = j
                    lowest_distance = distance
        next_row = remaining_cycle_list.pop(arg_min)
        end_cycle_list.append(next_row)
    return(end_cycle_list)

def hot_start_list_to_string(hot_start_list: list, locations: int, gray:bin) -> list:
    """invert the hot start integer list into a string
    
    Parameters:
    hot_start_list: list
        A list of integers showing the an estimate of the lowest cycle
    locations: int 
        The number of location in the problem
    gray:bin
        If True Gray codes are used

    Returns
    -------
    result_list: list
        A list of bits that represents the bit string for the lowest cycle
    
    """

    if len(hot_start_list) != locations:
        raise Exception(f'The hot start list should be length {locations}')
    
    first_item = hot_start_list.pop(0)
    #remove the first item for the list which should be zero
    if first_item != 0:
        raise Exception(f'The first item of the list must be zero')
    
    initial_list = [i for i in range(1, locations)]    
    total_binary_string = ''
    result_list = []
    
    for i, integer in enumerate(hot_start_list):
        bin_len = find_bin_length(len(initial_list))
        if bin_len > 0:
        #find the index of integer in hot start list
            index = initial_list.index(integer)
            if gray:
               binary_string = bin(graycode.tc_to_gray_code(index))
            else:
                binary_string = bin(index)
            binary_string = binary_string[2:] #remove the 0b charactor
            binary_string = binary_string.zfill(bin_len)
            total_binary_string += binary_string
            initial_list.pop(index)
    for i in range(len(total_binary_string)):
        result_list.append(int(total_binary_string[i]))
        #result_list.append(str(total_binary_string[i]))
    return(result_list)

def validate_gradient_type(gradient_type):
    """check that the gradient type is valid"""
    allowed_types = ['parameter_shift', 'SPSA']
    if gradient_type not in allowed_types:
        raise Exception (f'Gradient type {gradient_type} is not coded for')

def update_parameters_using_gradient(locations: int, iterations: int, 
                                     print_frequency: int, params: list, 
                                     rots: np.array, cost_fn, 
                                     qc: QuantumCircuit, shots: int, 
                                     s: float, eta: float, 
                                     average_slice: float, 
                                     gray: bool, 
                                     verbose: bool, 
                                     gradient_type: str='parameter_shift',
                                     alpha: float = 0.602,
                                     gamma:float = 0.101,
                                     c:float=1e-2,     
                                     ) -> list:
    """updates parameters using SPSA or parameter shift gradients"""
    cost_list, lowest_list, index_list, gradient_list = [], [], [], []
    parameter_list, average_list = [], []
    validate_gradient_type(gradient_type)

    if gradient_type == 'SPSA':
        define_parameters
        # A is <= 10% of the number of iterations normally, but here the number of iterations is lower.
        A = 50
    # order of magnitude of first gradients
        if verbose:
            print(f'c= {c}')

        abs_gradient = np.abs(my_gradient(cost_fn, qc, params, rots, s=s, 
                                          shots=shots, average_slice=average_slice,
                                          verbose=verbose,
                                          gradient_type='SPSA', ck=c,
                                          )
                              )
        
        magnitude_g0 = abs_gradient.mean()
        if verbose:
            print(f'abs_gradient {abs_gradient}, magnitude_g0 {magnitude_g0}')
            print(f'magnitude_g0, {magnitude_g0}')
 
    # 2 is an estimative of the initial changes of the parameters,
    # different changes might need other choices
        #a = 2*((A+1)**alpha)/magnitude_g0
        if magnitude_g0 == 0:
            a = 999
        else:
            a = 0.01*((A+1)**alpha)/magnitude_g0

    for i in range(0, iterations):
        bc = bind_weights(params, rots, qc)
        cost, lowest, lowest_energy_bit_string = cost_func_evaluate(cost_fn, bc, 
                                                                    shots, average_slice, 
                                                                    verbose)
        #cost is the top-sliced energy
        average, _ , _ = cost_func_evaluate(cost_fn, bc, shots, average_slice=1, verbose=verbose)
        #average is the average energy with no top slicing
        if verbose:
            print(f'cost, lowest, lowest_energy_bit_string = {cost}, {lowest}, {lowest_energy_bit_string}')
        if i == 0:
            lowest_string_to_date = lowest_energy_bit_string
            lowest_to_date = lowest
        else:
            if verbose:
                print(f'lowest,  lowest_to_date {lowest}, {lowest_to_date}')
            if lowest < lowest_to_date:
                if verbose:
                    print('Lowest less than lowest to date')
                lowest_to_date = lowest
                lowest_string_to_date = lowest_energy_bit_string
                if verbose:
                    print(f'lowest,  lowest_to_date {lowest}, {lowest_to_date}')
        route_list = convert_bit_string_to_cycle(lowest_string_to_date, locations, gray)
        index_list.append(i)
        cost_list.append(cost)
        lowest_list.append(lowest_to_date)
        average_list.append(average)
        parameter_list.append(rots)
        if gradient_type == 'parameter_shift':
            gradient = my_gradient(cost_fn, qc, params, rots, s, shots,
                                   average_slice=average_slice,
                                   verbose=verbose, gradient_type='parameter_shift'
                                   )
            
            rots = rots - eta * gradient
        elif gradient_type == 'SPSA':
            ak = a/((i+1+A)**(alpha))
            ck = c/((i+1)**(gamma))
            gradient = my_gradient(cost_fn, qc, params, rots, s, shots,
                                   average_slice=average_slice,
                                   verbose=verbose, gradient_type='SPSA',
                                   ck=ck
                                   )
            if verbose:
                print(f'For iteration {i} a = {a}, A = {A}, ak = {ak}, ck = {ck}')
                print(f'rots = {rots}')
                print(f'gradient = {gradient}')
            rots = rots - ak * gradient
        else:
            raise Exception(f'Error found when calculating gradient. {gradient_type} is not an allowed gradient type')
        gradient_list.append(gradient.tolist())
        if i % print_frequency == 0:  
            print(f'For iteration {i} using the best {average_slice*100} percent of the results')
            print(f'The average cost from the sample is {average:.3f} and the top-sliced average of the best results is {cost:.3f}')
            print(f'The lowest cost from the sample is {lowest:.3f}')
            print(f'The lowest cost to date is {lowest_to_date:.3f} corresponding to bit string {lowest_string_to_date} ')
            print(f'and route {route_list}')
            if verbose:
                print(f'The gradient is {gradient}')
                print(f'The rotations are {rots}')
    return index_list, cost_list, lowest_list, gradient_list, average_list, parameter_list
    
def cost_func_evaluate(cost_fn, bc: QuantumCircuit, 
                       shots: int = 1024, average_slice=1, 
                       verbose:bool=False) -> tuple:
    """evaluate cost function on a quantum computer
    
    Parameters
    ----------
    cost_fn: function
        A function of a bit string evaluating a distance for that bit string
    bc: QuantumCircuit
        A quantum circuit with bound weights for which the energy is to be found
    shots: int
        The number of shots for which the quantum circuit is to be run
    average_slice: float
        average over this slice of the energy.  For example:
        If average_slice = 1 then average over all energies.  
        If average_slice = 0.2 then average over the bottom 20% of energies
    verbose: bool
        If true outputs more data for debugging
    
    Returns
    -------
    cost: float
        The average cost evaluated
    lowest: float
        The lowest cost found
    lowest_energy_bit_string: string
        A list of the bits for the lowest energy bit string
    """

    sampler = SamplerV2()
    job = sampler.run([bc])
    results = job.result()
    counts = results[0].data.meas.get_counts()
    if verbose:
        print(f'The counts directory is {counts}')
    cost, lowest, lowest_energy_bit_string = find_stats(cost_fn, counts, shots, average_slice, verbose)
    return(cost, lowest, lowest_energy_bit_string)

def my_gradient(cost_fn, qc: QuantumCircuit, 
                params: list, rots: np.array, s: float=0.5, 
                shots:  int=1024, average_slice: float=1, verbose: bool=False,
                gradient_type: str='parameter_shift',
                ck:float=1e-2,     
                ) -> list:
    """calculate gradient for a quantum circuit with parameters and rotations
    
    Parameters
    ----------
    cost_fn: function
        A function of a bit string evaluating an energy (distance) for that bit string
    qc: QuantumCircuit
        A quantum circuit for which the gradient is to be found, without the weights being bound
    params: list
        A list of parameters (the texts)
    rots: list
        The exact values for the parameters, which are rotations of quantum gates
    s: float
        Determines the rotation for evaluating the gradient
    shots: int
        The number of shots for which the quantum circuit is to be run in each estimation of a parameter point
    average_slice: float
        Controls the amount of data to be included in the average.  
        For example, 0.2 means that the lowest 20% of distances found is included in the average.
    verbose: bool
        If True then more information is printed
    gradient_type: str
        controls the optimiser to be used.
        if 'parameter shift'  uses analytical expression
        if 'SPSA' uses a stochastical method

    Returns
    -------
    gradient:array
        The gradient for each parameter
    """
    new_rots = copy.deepcopy(rots)  
    if gradient_type == 'parameter_shift':
        gradient_list = []
        for i, theta in enumerate(rots):
            if verbose:
                print(f'processing {i}th weight')
                print(f'rot = {theta} i={i}')

            new_rots[i] = theta + np.pi/(4*s)

            if verbose:
                print(f'New rots+ = {new_rots}')
            bc = bind_weights(params, new_rots, qc)
            cost_plus, _, _ = cost_func_evaluate(cost_fn, bc, shots, 
                                                average_slice, verbose)
            
            new_rots[i] = theta - np.pi/(4*s)
            if verbose:
                print(f'New rots- = {new_rots}')
            bc = bind_weights(params, new_rots, qc)
            cost_minus, _, _ = cost_func_evaluate(cost_fn, bc, shots, 
                                                average_slice, verbose)

            delta = s * (cost_plus - cost_minus)

            if verbose:
                print(f'cost+ = {cost_plus} cost- = {cost_minus}, delta = {delta}')
            gradient_list.append(delta)
        gradient_array = np.array(gradient_list)

    elif gradient_type == 'SPSA':
        # number of parameters
        length = len(rots)
        # bernoulli-like distribution
        deltak = np.random.choice([-1, 1], size=length)
        
        # simultaneous perturbations
        ck_deltak = ck * deltak
        new_rots = rots + ck_deltak
        if verbose:
            print(f'New rots+ = {new_rots}')
           
        # gradient approximation]
        if verbose:
            print(f'params = {params}, new_rots = {new_rots}')
        bc = bind_weights(params, new_rots, qc)
        cost_plus, _, _ = cost_func_evaluate(cost_fn, bc, shots, 
                                             average_slice, verbose)

        new_rots = rots - ck_deltak
        if verbose:
            print(f'New rots- = {new_rots}')

        bc = bind_weights(params, new_rots, qc)
        cost_minus, _, _ = cost_func_evaluate(cost_fn, bc, shots, 
                                             average_slice, verbose)

        delta = cost_plus - cost_minus
        gradient_array = delta / (2 * ck_deltak)
        #need to return an array to match parameter shift
        if verbose:
            print(f'cost+ = {cost_plus} cost- = {cost_minus}, delta = {delta}') 
            print(f'gradient_array = {gradient_array}') 
            print(f'deltak = {deltak}')
            print(f'ck_deltak = {ck_deltak}')
    else:
        raise Exception(f'Gradient type {gradient_type} is not an allowed choice')
    return gradient_array

def define_parameters(qubits: int, mode: int=1) -> list:
    """set up parameters and initialise text
    
    Parameters
    ----------
    qubits: int
        The number of qubits in the circuit
    mode: int
        Controls setting the circuit up in different modes

    Returns
    -------
    params: list
        A list of parameters (the texts)

    """
    params = []
    if mode in [1,2]:
        num_params = 2 * qubits
        for i in range(num_params):
            text = "param " + str(i)
            params.append(Parameter(text))
        return params
    else:   
        raise Exception(f'Mode {mode} has not been coded for')

def vqc_circuit(qubits: int, params: list, mode:int=1) -> QuantumCircuit:
    """set up a variational quantum circuit

    Parameters
    ----------
    qubits: int
        The number of qubits in the circuit
    params: list
        A list of parameters (the texts)
    mode: int
        Controls setting the circuit up in different modes

    Returns
    -------
    qc: Quantum Circuit
        A quantum circuit without bound weights
    
    """

    qc = QuantumCircuit(qubits)
    if mode == 1:
        for i in range(qubits):
            qc.h(i)
            qc.ry(params[i], i)
            qc.rx(params[qubits+i], i)
        for i in range(qubits):
            if i < qubits-1:
                qc.cx(i,i+1)
            else:
                qc.cx(i,0)
                #ensure circuit is fully entangled
    elif mode == 2:
        for i in range(qubits):
            qc.rx(params[i], i)
        for i in range(qubits):
                if i < qubits-1:
                    qc.rxx(params[qubits+i], i, i+1,)
                else:
                    qc.rxx(params[qubits+i], i, 0,)
                #ensure circuit is fully entangled
    elif mode == 3:
    #test mode
        if qubits != 5:
            raise Exception(f'test mode {mode} is only to be used with 5 qubits.  {qubits} qubits are specified')
        qc.x(1)
        qc.x(3)
        qc.x(4)
    else:
        raise Exception(f'Mode {mode} has not been coded for')
    qc.measure_all()
    return qc

def create_initial_rotations(qubits: int, mode: int, bin_hot_start_list: list=[], 
                             hot_start: bool=False) -> list:
    """initialise parameters with random weights

    Parameters
    ----------
    qubits: int
        The number of qubits in the circuit
    mode: int
        Controls setting the circuit up in different modes
    hot_start: bool
        If true hot start values are used

    Returns
    -------
    init_rots: array
        initial rotations
    
    """
    if mode in [1,2]:
        param_num = 2 * qubits
    else:
        raise Exception(f'Mode {mode} is not yet coded')
    if hot_start:
        if mode in [1]:
            raise Exception('Cannot use a hot start for mode {mode}')
        init_rots = [0 for i in range(param_num)]
        for i, item in enumerate(bin_hot_start_list):
            if item == 1:
                init_rots[qubits-i-1] = np.pi 
                #need to reverse order because of qiskit convention
    else:
        init_rots= [random.random() * 2 * math.pi for i in range(param_num)]
    init_rots_array = np.array(init_rots)
    return(init_rots_array)

def bind_weights(params:list, rots:list, qc:QuantumCircuit) -> QuantumCircuit:
    """bind parameters to rotations and return a bound quantum circuit

    Parameters
    ----------
    params: list
        A list of parameters (the texts)
    rots: list
        The exact values for the parameters, which are rotations of quantum gates
    qc: Quantum Circuit
        A quantum circuit without bound weights  

    Returns
    -------
    bc: Quantum Circuit
        A quantum circuit with including bound weights, ready to run an evaluation
    """

    binding_dict = {}
    for i, rot in enumerate(rots):
        binding_dict[str(params[i])] = rot
    bc = qc.assign_parameters(binding_dict)
    return(bc)