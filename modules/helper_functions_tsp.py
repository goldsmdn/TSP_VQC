#helper functions for quantum circuit construction and evaluation
import math
import random
import copy
import graycode
import torch

from pathlib import Path

from braket.circuits import Circuit
from braket.parametric import FreeParameter
from braket.jobs.metrics import log_metric

from qiskit.circuit import Parameter
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeAuckland

from typing import Callable

from torch import mps # Import Callable for type hinting

from modules.helper_functions_general import (
    find_logical_to_physical_dictionary, 
    find_qubits_measured, 
    find_valid_device_loop,
    convert_physical_to_logical_bit_string,
    binary_string_format,
    load_dict_from_json,
    )

from classes.LRUCacheUnhashable import LRUCacheUnhashable

from modules.quantum_circuits import (
    mode_1,
    mode_2,
    mode_3,
    mode_4,
    mode_5,
    mode_6,
    mode_7,
    mode_13,
    )

from modules.config import (
    MODE_DISPATCH, 
    TARGETS,
    PRINT_FREQUENCY,
    NETWORK_DIR,
    DATA_SOURCES,
    )

import numpy as np

def find_bin_length(i: int) -> int:
    """find the length of a binary string to represent integer i"""
    if i <= 0:
        raise ValueError("n must be a positive integer")
    bin_len = math.ceil((math.log2(i)))
    return(bin_len) 

def read_file_name(
        locations: int, 
        data_sources: dict,
        file_type:str='file'
        ) -> str:
    """Find the filename for a certain number of locations
    
    Parameters
    ----------
    locations : int
        Number of locations, or vertices
    data_sources : dict
        Dictionary listing the filename for each problem size
    file_type : str
        Type of file to read - either 'file' or 'points'

    Returns
    ----------
    filename : string
        The filename for that problem size
    """
    
    if file_type == 'file':
        filename = data_sources[locations]['file']
        print(f'Reading distance data')
    elif file_type == 'points':
        filename = data_sources[locations]['points']
        print(f'Reading co-ordinate data')
    else:
        raise Exception(f'File type {file_type} is not coded for')
    return(filename)    

def validate_distance_array(array :np.ndarray, locs: int):
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

def find_distance(
        loc1: int, 
        loc2: int, 
        distance_array: np.ndarray,
        verbose: bool=False,
        ) -> float:
    """Finds the distance between locations using the distance matrix
    
    Parameters
    ----------
    loc1 : int
        First location
    loc2 : int
        Second location
    distance_array : np.ndarray
        An array containing the distances between locations
    verbose : bool
        If True then more information is printed

    Returns
    ----------
    distance : Float
        The distance between two locations
    """
    
    distance = distance_array[loc1][loc2]
    if verbose:
        print(f'The distance from location {loc1} to location {loc2} is {distance}')
    return(distance)

def find_problem_size(
    locations: int,
    formulation: str
) -> int:
    """Finds the number of binary variables needed
    
    Parameters
    ----------
    locations : int 
            Number of locations
    formulation: str
            'original' => method from Goldsmith D, Day-Evans J.
            'new' => method from Schnaus M, Palackal L, Poggel B, Runge X, Ehm H, Lorenz JM, et al.

    Returns
    ----------
    pb_dim : int
        Length of the bit string needed to store the problem
    """
    if formulation == 'original':
        pb_dim = 0
        for i in range(1, locations):
            bin_len = find_bin_length(i)
            pb_dim += bin_len
    elif formulation == 'new':
        f = math.factorial(locations)
        pb_dim = find_bin_length(f)
    else:
        raise ValueError(f'Unknown method {formulation}')
    return(pb_dim)

def convert_binary_list_to_integer(
        binary_list: list, 
        gray:bool=False)->int:
    """Converts list of binary numbers to an integer
    
    Parameters
    ----------
    binary_list : list
        List of binary numbers
    gray : bool
        Determines whether gray code is used.

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

def convert_integer_to_binary_list(
        integer: int, 
        length: int, 
        gray:bool=False)->list:
    """Converts an integer to a list of binary numbers
    
    Parameters
    ----------
    integer : int
        The integer to be converted
    length : int
        The length of the binary string
    gray : bool
        If True Gray codes are used

    Returns
    ----------
    result : list
        A list of binary numbers
    """
    if gray:
        integer = graycode.tc_to_gray_code(integer)
    binary_string = bin(integer)
    binary_string = binary_string_format(binary_string, length)
    result = [int(i) for i in binary_string]
    return(result)

def check_loc_list(loc_list:list, locs:int) -> bool:
    """Checks that the location list is a valid cycle with no repetition of nodes.
    
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
    """Completes the cycle by adding the missing location to the end of the cycle.
    
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

def find_total_distance(
        int_list: list, 
        locs: int, 
        distance_array :np.ndarray
        )-> float:
    """Finds the total distance for a valid formatted bit string representing a cycle.
    
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

def find_device_string(target):
    device_arn = TARGETS[target]['arn']
    return(device_arn)

def find_device(target):
    """
    Lazily create and return a Braket device.
    This function is hybrid-job safe.
    This code also work with qiskit
    """
    from braket.aws import AwsDevice
    from braket.devices import LocalSimulator

    cfg = TARGETS[target]
    # Local simulator
    if cfg['type'] == 'local_aws':
        return LocalSimulator()
    # AWS device
    device = AwsDevice(cfg['arn'])
    return device

def find_sdk(target:str) -> str:
    """Find the SDK (braket or qiskit) for a given target

    Parameters
    ----------
    target: str
        The target to find the sdk for.  This is a key in the TARGETS dictionary in config.py

    Returns
    -------
    sdk: str
        The sdk for the given target, either 'braket' or 'qiskit' that determines subsequent processing.
    """
    from modules.config import TARGETS
    return TARGETS[target]['sdk']

def find_type(target:str) -> str: 
    """Find the type (local_AWS, local_qiskit or aws) for a given target

    Parameters
    ----------
    target: str
        The target to find the type for.  This is a key in the TARGETS dictionary in config.py

    Returns
    -------
    type: str
        The type for  given target, that determines subsequent processing.
    """

def detect_quantum_GPU_support(target:str)-> bool:
    """Detect if a GPU is available for quantum simulations"""
    sdk_type = find_sdk(target)
    match sdk_type:
        case 'aws':
            #currently no GPU support for AWS simulators, but could be added in the future
            return False
        case 'qiskit':
            devices = AerSimulator().available_devices()
            if 'GPU' in devices:
                return True
            else:
                return False
        case _:
            raise Exception(f'SDK {sdk_type} has not been coded for')
    
def bind_weights(
        params:list, 
        rots:list, 
        qc:Circuit,
        target:str) -> Circuit:
    """Bind parameters to rotations and return a bound quantum circuit

    Parameters
    ----------
    params: list
        A list of parameters (the texts)
    rots: list
        The exact values for the parameters, which are rotations of quantum gates
    qc: Circuit
        A quantum circuit without bound weights  
    target: str
        This is a key in the TARGETS dictionary

    Returns
    -------
    bc: Quantum Circuit
        A quantum circuit with including bound weights, ready to run an evaluation
    """
    circuit_sdk = find_sdk(target)
    binding_dict = {}
    for i, rot in enumerate(rots):
        param_name = str(params[i])
        binding_dict[param_name] = rot
    match circuit_sdk:
        case 'aws':
            bc = qc.make_bound_circuit(binding_dict)
        case 'qiskit':
            bc = qc.assign_parameters(binding_dict)
        case _:
            raise Exception(f'SDK {circuit_sdk} has not been coded for')
    return(bc)

def define_parameters(            
        mode:int, 
        num_params:int,
        target:str) -> list:
    """Set up parameters and initialise text
    
    Parameters
    ----------
    #qubits: int - The number of qubits in the circuit
    mode: int - Controls setting the circuit up in different modes
    num_params: int - The number of parameters to be defined
    target: str - This is a key in the TARGETS dictionary

    Returns
    -------
    params: list
        A list of parameters (the texts)

    """
    circuit_sdk = find_sdk(target)
    params = []

    for i in range(num_params):
        text = "param_" + str(i)
        match circuit_sdk:
            case 'aws':
                params.append(FreeParameter(text))
            case 'qiskit':
                params.append(Parameter(text))
            case _:
                raise Exception(f'Mode {mode} has not been coded for')
    return params
    
def vqc_circuit(
        qubits: int,
        mode:int,
        noise_bool:bool,
        layers:int,
        params:list,
        target:str) -> Circuit:
    """Set up a variational quantum circuit

    Parameters
    ----------
    A sub data logger holding the parameters for the run with key fields:
    qubits: int
        The number of qubits in the circuit
    mode: int
        Controls setting the circuit up in different modes
    noise: bool
        Controls if noise is included in the circuit
    layers: int
        The numnber of layers
    params: list
        A list of parameters (the texts)

    Returns
    -------
    qc: Quantum Circuit
        A quantum circuit without bound weights
    """

    circuit_sdk = find_sdk(target)
    qubit_dict = find_logical_to_physical_dictionary(qubits, target)
    qubits_measured = find_qubits_measured(qubits, target)
    
    context_dict = {
        'qubits': qubits,
        'params': params,
        'layers': layers,   
        'qubit_dict': qubit_dict,
        'qubits_measured': qubits_measured,
        }
    
    qc = MODE_DISPATCH[mode]['circuit'](context_dict)
        
    # only measure the qubits in the sorted list
    valid_device_loop = find_valid_device_loop(qubits, target)
    sorted_list = sorted(valid_device_loop)
    match circuit_sdk:
        case 'aws':
            qc.measure(sorted_list)
        case 'qiskit':            
            qc.measure(sorted_list, sorted_list)
        case _:
            raise Exception(f'SDK {circuit_sdk} has not been coded for')
    print(f'After measurement, the following qubits are measured {sorted_list}') 

    if noise_bool:
        backend = FakeAuckland()
        qc = transpile(qc, backend=backend)
    return qc

#def define_parameters(                
#        mode:int, 
#        num_params:int
#        ) -> list:
    """Set up parameters and initialise text
    
    Parameters
    ----------
        mode: int - Controls setting the circuit up in different modes

    Returns
    -------
    params: list
        A list of parameters (the texts)

    """
    """params = []
    circuit_sdk = MODE_DISPATCH[mode]['sdk']
         
    for i in range(num_params):
        text = "param_" + str(i)
        match circuit_sdk:
            case 'aws':
                params.append(FreeParameter(text))
            case 'qiskit':
                params.append(Parameter(text)) 
            case _:
                raise Exception(f'Mode {mode} has not been coded for')
    return params"""

def create_initial_rotations(
        qubits: int,
        num_params: int,
        target:str,
        hot_start:bool=False,
        bin_hot_start_list: list=False,)-> np.ndarray: 
    """Initialise parameters with random weights, or hot start list

    Parameters
    ----------
    qubits : int
        The number of qubits in the circuit
    mode : int
        Controls setting the circuit up in different modes
    layers : int
        The number of layers
    target : str
        The target quantum device
    hot_start : bool
        If true hot start values are used
    bin_hot_start_list : list
        Binary list containing the hot start values

    Returns
    -------
    init_rots: array
        initial rotations
    
    """
    circuit_sdk = find_sdk(target)
    if hot_start:
        init_rots = [0 for i in range(num_params)]
        for i, item in enumerate(bin_hot_start_list):
            if item == 1:
                match circuit_sdk:
                    case 'aws':
                        init_rots[i] = np.pi 
                    case 'qiskit':  
                        init_rots[qubits-i-1] = np.pi 
                #need to reverse order because of qiskit convention
    elif not hot_start:
        init_rots= [random.random() * 2 * math.pi for i in range(num_params)]
    else:
        raise Exception('Hot_start must be a boolean')
    init_rots_array = np.array(init_rots)
    return(init_rots_array)

def cost_func_evaluate(
        noise_bool:bool,
        shots:int,
        cost_fn:Callable,
        model,
        target,
        mps,
        average_slice:float=1,
        ) -> tuple[float, float, list[int]]:             
                       
    """Evaluate cost function on a quantum computer
    
    Parameters
    ----------
    noise: bool
        If True a noisy quantum computer is used
    quantum: bool
        If True a quantum computer is used.  If False a classical model is used
    shots: int
        The number of shots for which the quantum circuit is to be run  
    cost_fn: function
        A function of a bit string evaluating a distance for that bit string
    model : a model to evalulate an output bit string given weights eg
        Circuit
            A quantum circuit with bound weights for which the energy is to be found
        Classical Model
            A classical model with bound weights for which the energy is to be found
    average_slice: float
        average over this slice of the energy.  For example:
        If average_slice = 1 then average over all energies.  
        If average_slice = 0.2 then average over the bottom 20% of energies
    
    Returns
    -------
    cost: float
        The average cost evaluated
    lowest: float
        The lowest cost found
    lowest_energy_bit_string: string
        A list of the bits for the lowest energy bit string
    """

    sdk_type = find_sdk(target)
    match sdk_type:
        case 'aws':
            device = find_device(target)
            job = device.run(model, shots=shots)    
            result = job.result()
            counts = result.measurement_counts 
        case 'qiskit':
            if noise_bool:
                backend = FakeAuckland()
                job = backend.run(model, shots=shots)     
                counts = job.result().get_counts()
            elif mps:
                simulator = AerSimulator(method='matrix_product_state')
                results = simulator.run(model).result()
                counts = results.get_counts(model)
            else:
                if detect_quantum_GPU_support(target):
                    simulator = AerSimulator(
                        method='statevector', 
                        device='GPU'
                        )
                    results = simulator.run(model).result()
                    counts = results.get_counts(model)
                else:
                    simulator = AerSimulator(method='statevector')
                    results = simulator.run(model).result()
                    counts = results.get_counts(model)
        case _:
            raise Exception(f'SDK {sdk_type} has not been coded for')

    cost, lowest, lowest_energy_bit_string = find_stats(
        cost_fn=cost_fn, 
        counts=counts, 
        shots=shots, 
        average_slice=average_slice
    )
    return(cost, lowest, lowest_energy_bit_string)

def is_even(n):
    return n % 2 == 0

def validate_qubit_loops(qubits, loop_dict, target):
    device = find_device(target=target)
    print(f'Found device as {device}')
    if not hasattr(device, "properties"):
        #handle local emulators
        print("Local device detected — skipping connectivity check.")
        return
    else:
        connectivity_dict = device.properties.dict()['paradigm']['connectivity']
        loop_list = loop_dict[target][qubits]
        set_list = set(loop_list)
        if len(set_list) != len(loop_list):
            raise Exception(f'The loop list {loop_list} contains duplicates')
        for index, qubit in enumerate(loop_list):
            last_valid_index = len(loop_list) - 1
            if index < last_valid_index:
                next_qubit = loop_list[index+1]
            else:
                next_qubit = loop_list[0]
            connected_qubits = connectivity_dict['connectivityGraph'][str(qubit)]
            if str(next_qubit) not in connected_qubits:
                raise Exception(f'qubits{qubit} and {next_qubit} are not connected')
        loop_length = len(loop_list)
        if is_even(qubits) and loop_length != qubits:
            raise Exception(f'{len(loop_list)=} and {qubits=} for {target=}')
        if not is_even(qubits) and (loop_length - qubits) !=1:
            raise Exception(f'{len(loop_list)=} and {qubits=} for {target=}')
        print(f'No errors found for {target=} {qubits=} \n')
    return

def find_stats(
        cost_fn: Callable,
        counts: dict, 
        shots: int, 
        average_slice: float=1, 
        )-> tuple[float, float, list[int]]:
    """Finds the average energy of the relevant counts, and the lowest energy
    
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
        total_counts += count
        total_energy += energy * count

    if shots != total_counts:
        raise Exception(f'The total_counts {total_counts=} does not agree to the {shots=}')

    if slicing:
        accum, total_energy, total_counts = 0, 0, 0
        stop = shots * average_slice
        sorted_energy_dict = dict(sorted(energy_dict.items()))
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

    return(average_energy, lowest_energy, lowest_energy_bit_string)

def update_parameters_using_gradient(
        locations:int,
        qubits:int,
        average_slice:float,
        shots:int,
        mode:int,
        iterations:int,
        gray:bool,
        hot_start:bool,
        gradient_type:str,
        formulation:str,
        layers:int,
        alpha:float,     
        big_a:float,
        c:float,
        eta:float,
        gamma:float,
        s:float,
        noise_bool:bool,                                 
        params: list,
        rots: np.ndarray,
        cost_fn: Callable,
        qc: Circuit,
        target: str,
        mps: bool,
        print_results: str=True,
        ):
    """Updates parameters using SPSA or parameter shift gradients"""
    cost_list, lowest_list, index_list, gradient_list = [], [], [], []
    parameter_list, average_list = [], []

    validate_gradient_type(gradient_type)

    if gradient_type == 'SPSA':
        #define_parameters
        # A is <= 10% of the number of iterations normally, but here the number of iterations is lower.
        # order of magnitude of first gradients

        if any(x is None for x in (s, eta, big_a, alpha)):
            raise Exception(f'{s=}, {eta=}, {big_a=}, {alpha=}')
        
        abs_gradient = np.abs(my_gradient(
            noise_bool=noise_bool,
            shots=shots,
            s=s,
            gradient_type=gradient_type,
            cost_fn=cost_fn, 
            qc=qc, 
            params=params, 
            rots=rots, 
            average_slice=average_slice,
            target=target,
            mps=mps,
            )
        )
        
        magnitude_g0 = abs_gradient.mean()
 
        if magnitude_g0 == 0:
        # stop div by zero error
            a = 999
        else:
            a = eta*((big_a+1)**alpha)/magnitude_g0

    for i in range(0, iterations):
        bc = bind_weights(
            params=params, 
            rots=rots, 
            qc=qc,
            target=target,
            )
        cost, lowest, lowest_energy_bit_string = cost_func_evaluate(
            noise_bool=noise_bool,
            shots=shots,
            cost_fn=cost_fn, 
            model=bc,
            target=target,
            mps=mps,
            average_slice=average_slice, 
            )
        #cost is the top-sliced energy
        average, _ , _ = cost_func_evaluate(
            noise_bool=noise_bool,
            shots=shots,
            cost_fn=cost_fn, 
            model=bc, 
            target=target,
            mps=mps,
            average_slice=1, 
            )
        #average is the average energy with no top slicing
        if i == 0:
            lowest_string_to_date = lowest_energy_bit_string
            lowest_to_date = lowest
        else:
            if lowest < lowest_to_date:
                lowest_to_date = lowest
                lowest_string_to_date = lowest_energy_bit_string
        lowest_string_to_date = convert_physical_to_logical_bit_string(lowest_string_to_date, qubits, target)
        route_list = convert_bit_string_to_cycle(
            lowest_string_to_date, 
            locations, 
            gray, 
            formulation,
            )
        index_list.append(i)
        cost_list.append(cost)
        lowest_list.append(lowest_to_date)
        average_list.append(average)
        parameter_list.append(rots)
        if gradient_type == 'parameter_shift':
            gradient = my_gradient(
                noise_bool=noise_bool,
                shots=shots,
                s=s,
                gradient_type=gradient_type,
                cost_fn=cost_fn, 
                qc=qc, 
                params=params, 
                rots=rots,
                average_slice=average_slice,
                target=target,
                mps=mps,
                ck=ck,
                )
            
            rots = rots - eta * gradient
        elif gradient_type == 'SPSA':
            ak = a/((i+1+big_a)**(alpha))
            ck = c/((i+1)**(gamma))
            gradient = my_gradient(
                noise_bool=noise_bool,
                shots=shots,
                s=s,
                gradient_type=gradient_type,
                cost_fn=cost_fn, 
                qc=qc, 
                params=params, 
                rots=rots, 
                average_slice=average_slice,
                target=target,
                mps=mps,
                ck=ck,
                )
            rots = rots - ak * gradient
        else:
            raise Exception(f'Error found when calculating gradient. {gradient_type} is not an allowed gradient type')
        gradient_list.append(gradient.tolist())
        if print_results and i % PRINT_FREQUENCY == 0:  
            #Force flush to push to Cloudwatch quickly
            print(f'For iteration {i} using the best {average_slice*100} percent of the results', flush=True)
            print(f'The average cost from the sample is {average:.3f} and the top-sliced average of the best results is {cost:.3f}', flush=True)
            print(f'The lowest cost from the sample is {lowest:.3f}', flush=True)
            print(f'The lowest cost to date is {lowest_to_date:.3f} corresponding to bit string {lowest_string_to_date}', flush=True)
            print(f'and route {route_list}')
            #AWS hybrid job
            log_metric(metric_name="average_sample_cost", iteration_number=i, value=average)
            log_metric(metric_name="top_sliced_sample_cost", iteration_number=i, value=cost)
            log_metric(metric_name="lowest_sample_cost", iteration_number=i, value=lowest)
            log_metric(metric_name="lowest_to_date", iteration_number=i, value=lowest_to_date)
                
    return index_list, cost_list, lowest_list, gradient_list, average_list, parameter_list 
    
def my_gradient(
    noise_bool:bool,
    shots:int,
    s:float,
    gradient_type:str,
    cost_fn,
    qc:Circuit,
    params:list, 
    rots:np.ndarray, 
    average_slice:float,
    target:str, 
    mps:bool,
    ck:float=1e-2,
    ) -> np.ndarray:
    """Calculate gradient for a quantum circuit with parameters and rotations
    
    Parameters
    ----------
    noise: bool
        If True a noisy quantum computer is used
    shots: int
        The number of shots for which the quantum circuit is to be run  
    s: float
        Parameter shift parameter
    gradient_type: str
        controls the optimiser to be used.
        if 'parameter shift'  uses analytical expression
        if 'SPSA' uses a stochastical method
    ck: float
        SPSA parameter, small number controlling perturbations
    cost_fn: function
        A function of a bit string evaluating an energy (distance) for that bit string
    qc: Circuit
        A quantum circuit for which the gradient is to be found, without the weights being bound
    params: list
        A list of parameters (the texts)
    rots: list
        The exact values for the parameters, which are rotations of quantum gates
    average_slice: float
        Controls the amount of data to be included in the average.  
        For example, 0.2 means that the lowest 20% of distances found is included in the average.

    Returns
    -------
    gradient:array
        The gradient for each parameter
    """
    new_rots = copy.deepcopy(rots)  
    if gradient_type == 'parameter_shift':
        gradient_list = []
        for i, theta in enumerate(rots):
            new_rots[i] = theta + np.pi/(4*s)
            bc = bind_weights(
                params=params, 
                rots=new_rots, 
                qc=qc,
                target=target,
                )
            cost_plus, _, _ = cost_func_evaluate(
                noise_bool=noise_bool,
                shots=shots,
                cost_fn=cost_fn, 
                model=bc, 
                target=target,
                mps=mps,
                average_slice=average_slice, 
                )
            
            new_rots[i] = theta - np.pi/(4*s)
            bc = bind_weights(
                params=params, 
                rots=new_rots, 
                qc=qc,
                target=target,
                )
            cost_minus, _, _ = cost_func_evaluate(
                noise_bool=noise_bool,
                shots=shots,
                cost_fn=cost_fn, 
                model=bc, 
                target=target,
                mps=mps,
                average_slice=average_slice, 
                )
            delta = s * (cost_plus - cost_minus)
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
        
        # gradient approximation
        bc = bind_weights(
            params=params, 
            rots=new_rots,
            qc=qc,
            target=target,
            )
        cost_plus, _, _ = cost_func_evaluate(
            noise_bool=noise_bool,
            shots=shots,
            cost_fn=cost_fn, 
            model=bc, 
            target=target,
            mps=mps,
            average_slice=average_slice, 
            )

        new_rots = rots - ck_deltak
        bc = bind_weights(
            params=params, 
            rots=new_rots,
            qc=qc,
            target=target,
            )
        cost_minus, _, _ = cost_func_evaluate(
            noise_bool=noise_bool,
            shots=shots,
            cost_fn=cost_fn, 
            model=bc, 
            target=target,
            mps=mps,
            average_slice=average_slice, 
            )

        delta = cost_plus - cost_minus
        gradient_array = delta / (2 * ck_deltak)
        #need to return an array to match parameter shift
    else:
        raise Exception(f'Gradient type {gradient_type} is not an allowed choice')
    return gradient_array   

def validate_gradient_type(gradient_type):
    """Check that the gradient type is valid"""
    allowed_types = ['parameter_shift', 'SPSA']
    if gradient_type not in allowed_types:
        raise Exception (f'Gradient type {gradient_type} is not coded for')
    
def convert_bit_string_to_cycle(
        bit_string: list, 
        locs: int, 
        gray: bool=False, 
        method: str='original') -> list:
    """Converts a bit string to a cycle.
    
    Parameters
    ----------
    bit_string : list
        A list of zeros and ones produced by the quantum computer
    locs: int
        The number of locations
    gray: bool
        If True Gray codes are used
    method: str
        'original' => method from Goldsmith D, Day-Evans J. 
        'new' => method from Schnaus M, Palackal L, Poggel B, Runge X, Ehm H, Lorenz JM, et al.

    Returns
    ----------
    end_cycle_list : list
        A list of integers showing a cycle.  The bit string is processed from left to right
    """

    end_cycle_list = []
    start_cycle_list = [i for i in range(locs)]

    if method == 'original':
        #need to avoid changing the original bit_string
        bit_string_copy = copy.deepcopy(bit_string)
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
    elif method == 'new':
        f = math.factorial(locs)
        bit_string_length = math.ceil(math.log2(f))
        if len(bit_string) != bit_string_length:
            raise Exception(f'bit_string length {len(bit_string)} does not match {bit_string_length}')
        x = convert_binary_list_to_integer(bit_string, gray=gray)
        y = x % f
        i = 0
        while i < locs:
            f = int(f / (locs - i))
            #correcting mistype in paper
            k = math.floor(y / f)
            end_cycle_list.append(start_cycle_list[k])
            start_cycle_list.remove(start_cycle_list[k])
            #correcting mistype in paper
            y -= k * f
            i += 1
        return end_cycle_list
    else:
        raise Exception(f'Unknown method {method}')
    
def cost_fn_fact(
    locations:int,
    qubits,
    gray:bool, 
    formulation:str, 
    distance_array: np.ndarray, 
    target: str
    ) -> Callable[[list], int]:    
    """ Returns a cost function inside a decorator,

    Parameters
    ----------
    locations: int
        The number of locations in the problem
    qubits:int
        The number of qubits required
    gray: bool
        If True Gray codes are used
    formulation: str
        'original' => method from Goldsmith D, Day-Evans J.
        'new' => method from Schnaus M, Palackal L, Poggel B, Runge X, Ehm H, Lorenz JM, et al.
    distance_array: array
        Numpy symmetric array with distances between locations
    
    Returns
    -------
    cost_fn: cost function
        A function of a bit string evaluating a distance for that bit string
    
    """
    @LRUCacheUnhashable
    def cost_fn(bit_string_input: list) -> float:
        """Returns the value of the objective function for a bit_string"""
        if isinstance(bit_string_input, list):
            if target == 'ml':
                #no need to convert from physical to logical bit string as 
                #the input is not from a quantum computer
                bit_string = bit_string_input
            else:
                bit_string = convert_physical_to_logical_bit_string(
                    input_bitstring = bit_string_input, 
                    qubits=qubits, 
                    target=target
                    )
            full_list_of_locs = convert_bit_string_to_cycle(
                bit_string=bit_string,
                locs=locations,
                gray=gray,
                method=formulation
            )
            total_distance = find_total_distance(
                int_list=full_list_of_locs, 
                locs=locations, 
                distance_array=distance_array
                )
            valid = check_loc_list(
                loc_list=full_list_of_locs,
                locs=locations
                )
            if not valid:
                raise Exception('Algorithm returned incorrect cycle')  
            return total_distance
        else:
            raise Exception(f'bit_string {bit_string_input} is not a list or a tensor')
    return cost_fn

def cost_fn_tensor(input: torch.tensor, 
                   cost_fn: Callable)-> torch.Tensor:

    """ Find the distance for each bit string input using cost_fn

    Parameters
    ----------
    input : torch.tensor
        Torch array with n bit strings for analysis
    cost_fn : function
        maps a bit_list to a distance
    
    Returns
    -------
    distance_tensor : torch.tensor
        a Torch array with one distance entry for each input

    """

    if isinstance(input, torch.Tensor):
        if input.dim() != 2:
            raise Exception(f'input= {input} is a Torch tensor but does not have dimension 2')
        rows = input.size(0)
        distance_tensor = torch.zeros(rows)
        for i in range(rows):
            row = input[i]
            bit_string = row.int().tolist()
            distance = cost_fn(bit_string)
            distance_tensor[i] = distance
        return distance_tensor
    else:
        raise Exception(f'bit_string {input} is not a tensor')
    
def hot_start_list_find(
        locations: int, 
        distance_array: np.ndarray) -> list:
    """Finds a route from a distance array where the distance to the next point is the shortest available
    
    Parameters
    ----------
    locations : int
        The number of locations in the problem
    distance_array: array
        Numpy symmetric array with distances between locations

    Returns
    -------
    end_cycle_list: list
        A list of integers showing the an estimate of the lowest cycle
    
    """
    validate_distance_array(distance_array, locations)
    remaining_cycle_list = [i for i in range(locations)]
    end_cycle_list = []
    end_cycle_list.append(remaining_cycle_list.pop(0)) #start point of cycle is always 0
    next_row = 0
    for i in range(locations-1):
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

def hot_start_list_to_string(
    locations: int,
    gray: bool,
    formulation: str,
    hot_start_list: list) -> list:
    """Invert the hot start integer list into a string
    
    Parameters:
    -----------

    locations: int
        The number of locations in the problem
    gray: bool
        If True Gray codes are used
    formulation: str
        The formulation to use
    hot_start_list: list
        A list of integers showing an estimate of the lowest cycle

    Returns
    -------
    result_list: list
        A list of bits that represents the bit string for the lowest cycle
        sdl.formulation: str
            'original' => method from Goldsmith D, Day-Evans J.
            'new' => method from Schnaus M, Palackal L, Poggel B, Runge X, Ehm H, Lorenz JM, et al.
    hot_start_list: list
        A list of integers showing an estimate of the lowest cycle

    Returns
    -------
    result_list: list
        A list of bits that represents the bit string for the lowest cycle
    
    """
    
    if formulation == 'original':
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
                binary_string = binary_string_format(binary_string, bin_len)
                total_binary_string += binary_string
                initial_list.pop(index)
        for i in range(len(total_binary_string)):
            result_list.append(int(total_binary_string[i]))
        return(result_list)
    elif formulation == 'new':
        dim = find_problem_size(
            locations=locations,
            formulation=formulation
        )    
        f = math.factorial(locations)
        y = 0
        i = 0
        start_cycle_list = [i for i in range(locations)]
        while i < locations:
            f = int(f / (locations - i))
            m = hot_start_list[i]
            j = start_cycle_list.index(m)
            start_cycle_list.remove(m)  
            y += j * f
            i += 1

        result_list = convert_integer_to_binary_list(y, dim, gray=gray)
        return result_list
    else:
        raise Exception(f'Unknown method {formulation}')

def find_run_stats(lowest_list:list)-> tuple[float, int]:
    """Finds the lowest energy and the iteration at which it was found
    
    Parameters
    ----------
    lowest_list: list
        A list of floats showing the lowest energy found at that iteration

    Returns
    -------
    lowest_energy: float
        The lowest energy found
    iteration: int
        The iteration at which the lowest energy was found
    """
    previous_lowest = max(lowest_list)
    lowest_energy = previous_lowest
    iteration = 0
    for i, value in enumerate(lowest_list):
        if value <  previous_lowest:
            lowest_energy = value
            previous_lowest = value
            iteration = i
    return lowest_energy, iteration

def find_distances_array(locations:int, 
                         print_comments:bool=False)-> tuple[np.ndarray, float]:
    """Finds the array of distances between locations and the best distance"""
    sources_filename = Path(NETWORK_DIR).joinpath(DATA_SOURCES)
    data_source_dict = load_dict_from_json(sources_filename)
    filename = read_file_name(str(locations), data_source_dict)
    filepath = Path(NETWORK_DIR).joinpath(filename)
    best_dist = data_source_dict[str(locations)]['best']
    if print_comments:
        print(f'Data will be read from filename {filepath}.') 
        print(f'It is known that the shortest distance is {best_dist}')
    distance_array = np.genfromtxt(filepath)
    validate_distance_array(distance_array, locations)
    return distance_array, best_dist 

def calculate_hot_start_data(
    sdl, 
    distance_array: np.ndarray, 
    cost_fn: Callable,
    print_results:bool=False,
    )-> tuple[list, float]:
    """Calculate hot start data from a distance array
    
    Parameters
    ----------
    sdl : SubDataLogger object 
        Containing key parameters
    distance_array: array
        Numpy symmetric array with distances between locations
    cost_fn : function  
        A function that returns a distance from a binary string
    print_results: bool
        If true prints out the hot start data

    Returns
    -------
    hot_start_list: list
        A list of integers showing an estimate of the lowest cycle
    hot_start_distance: float
        The distance of the hot start cycle

    """
    hot_start_list = hot_start_list_find(
        locations=sdl.locations, 
        distance_array=distance_array, 
        )
    #bin_hot_start_list =  hot_start_list_to_string(sdl, hot_start_list)
    bin_hot_start_list = hot_start_list_to_string(
        locations=sdl.locations,
        gray=sdl.gray,
        formulation=sdl.formulation,
        hot_start_list=hot_start_list,
    )
    hot_start_distance = cost_fn(bin_hot_start_list)
    if print_results:
        print(f'The hot start location list is {hot_start_list}')
        print(f'This is equivalent to a binary list: {bin_hot_start_list}')
        print(f'The hot start distance is {hot_start_distance}, compared to a best distance of {sdl.best_dist}.')
        print(f'The hot start distance is {hot_start_distance}')
    return bin_hot_start_list, hot_start_distance      