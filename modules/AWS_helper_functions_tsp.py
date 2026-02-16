import numpy as np
import math
import copy
import graycode

from braket.parametric import FreeParameter
from braket.circuits import Circuit
from braket.devices import Devices, LocalSimulator
from braket.aws import AwsDevice

import random
from typing import Callable # Import Callable for type hinting

from classes.LRUCacheUnhashable import LRUCacheUnhashable

from modules.helper_functions_tsp import (convert_bit_string_to_cycle, 
                                  find_total_distance,
                                  find_bin_length,
                                  check_loc_list,
                                  validate_distance_array,
                                  binary_string_format,
                                  convert_integer_to_binary_list,
                                  validate_gradient_type
                                  ) 


from braket.jobs.metrics import log_metric

from modules.config import PRINT_FREQUENCY

def cost_fn_fact(locations:int, 
                 gray:bool, 
                 formulation:str, 
                 distance_array: np.ndarray, ) -> Callable[[list], int]:
    """ Returns a cost function inside a decorator,

    Parameters
    ----------
    locations: int
        The number of locations in the problem
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
            bit_string = bit_string_input
            full_list_of_locs = convert_bit_string_to_cycle(bit_string, 
                                                            locations, 
                                                            gray, 
                                                            formulation
                                                            )
            total_distance = find_total_distance(full_list_of_locs, 
                                                 locations, 
                                                 distance_array
                                                 )
            valid = check_loc_list(full_list_of_locs,
                                   locations
                                   )
            if not valid:
                raise Exception('Algorithm returned incorrect cycle')  
            return total_distance
        else:
            raise Exception(f'bit_string {bit_string_input} is not a list or a tensor')
    return cost_fn

def find_stats(cost_fn: Callable,
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

def hot_start(locations:int, distance_array: np.ndarray) -> list:
    """Finds a route from a distance array where the distance to the next point is the shortest available
    
    Parameters
    ----------
    locations: int
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
    
def hot_start_list_to_string(locations:int, 
                             hot_start_list:list, 
                             gray:bool=False,
                             formulation:str='original') -> list:
    """Invert the hot start integer list into a string
    
    Parameters:
    -----------

    locations: int 
        The number of location in the problem
    gray: bool
        If True Gray codes are used
    formulation: str
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
            raise Exception(f'The first item of the list is {first_item} and should be zero')
        
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
        dim = find_problem_size(locations, formulation)
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

def update_parameters_using_gradient(locations:int,
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
                                     noise:bool,                                 
                                     params: list,
                                     rots: np.ndarray,
                                     cost_fn: Callable,
                                     qc: Circuit,
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
        
        abs_gradient = np.abs(my_gradient(noise,
                                          shots,
                                          s,
                                          gradient_type,
                                          cost_fn, 
                                          qc, 
                                          params, 
                                          rots, 
                                          average_slice,
                                          )
                                )
        
        magnitude_g0 = abs_gradient.mean()
 
        if magnitude_g0 == 0:
        # stop div by zero error
            a = 999
        else:
            a = eta*((big_a+1)**alpha)/magnitude_g0

    for i in range(0, iterations):
        bc = bind_weights(params, rots, qc)
        cost, lowest, lowest_energy_bit_string = cost_func_evaluate(noise,
                                                                    shots,
                                                                    cost_fn, 
                                                                    bc, 
                                                                    average_slice=average_slice, 
                                                                    )
        #cost is the top-sliced energy
        average, _ , _ = cost_func_evaluate(noise,
                                            shots,
                                            cost_fn, 
                                            bc, 
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
        route_list = convert_bit_string_to_cycle(lowest_string_to_date, 
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
            gradient = my_gradient(noise,
                                   shots,
                                   s,
                                   gradient_type,
                                   cost_fn, 
                                   qc, 
                                   params, 
                                   rots, 
                                   average_slice=average_slice,
                                   )
            
            rots = rots - eta * gradient
        elif gradient_type == 'SPSA':
            ak = a/((i+1+big_a)**(alpha))
            ck = c/((i+1)**(gamma))
            gradient = my_gradient(noise,
                                   shots,
                                   s,
                                   gradient_type,
                                   cost_fn, 
                                   qc, 
                                   params, 
                                   rots, 
                                   average_slice=average_slice,
                                   ck=ck,
                                   )
            rots = rots - ak * gradient
        else:
            raise Exception(f'Error found when calculating gradient. {gradient_type} is not an allowed gradient type')
        gradient_list.append(gradient.tolist())
        #print(f'Ready to print results with {PRINT_FREQUENCY=} and {print_results=}', flush=True)
        if print_results and i % PRINT_FREQUENCY == 0:  
            #Force flush to push to Cloudwatch quickly
            print(f'For iteration {i} using the best {average_slice*100} percent of the results', flush=True)
            print(f'The average cost from the sample is {average:.3f} and the top-sliced average of the best results is {cost:.3f}, flush=True')
            print(f'The lowest cost from the sample is {lowest:.3f}, flush=True')
            print(f'The lowest cost to date is {lowest_to_date:.3f} corresponding to bit string {lowest_string_to_date}, flush=True ')
            print(f'and route {route_list}')
            #AWS hybrid job
            log_metric(metric_name="average_sample_cost", iteration_number=i, value=average)
            log_metric(metric_name="top_sliced_sample_cost", iteration_number=i, value=cost)
            log_metric(metric_name="lowest_sample_cost", iteration_number=i, value=lowest)
            log_metric(metric_name="lowest_to_date", iteration_number=i, value=lowest_to_date)
                
    return index_list, cost_list, lowest_list, gradient_list, average_list, parameter_list 
    
def cost_func_evaluate(noise,
                       shots,
                       cost_fn:Callable,
                       model,
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

    #device = LocalSimulator()
    device = AwsDevice(Devices.Amazon.SV1)
        # Run on Braket
    job = device.run(model, shots=shots)    
    result = job.result()
    counts = result.measurement_counts 
    
    cost, lowest, lowest_energy_bit_string = find_stats(cost_fn, counts, shots, average_slice, )
    return(cost, lowest, lowest_energy_bit_string)

def my_gradient(noise:bool,
                shots:int,
                s:float,
                gradient_type:str,
                cost_fn,
                qc:Circuit,
                params:list, 
                rots:np.ndarray, 
                average_slice:float, 
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
            bc = bind_weights(params, new_rots, qc)
            cost_plus, _, _ = cost_func_evaluate(noise,
                                                 shots,
                                                 cost_fn, 
                                                 bc, 
                                                 average_slice, 
                                                 )
            
            new_rots[i] = theta - np.pi/(4*s)
            bc = bind_weights(params, new_rots, qc)
            cost_minus, _, _ = cost_func_evaluate(noise,
                                                  shots,
                                                  cost_fn, 
                                                  bc, 
                                                  average_slice, 
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
        bc = bind_weights(params, new_rots, qc)
        cost_plus, _, _ = cost_func_evaluate(noise,
                                             shots,
                                             cost_fn, 
                                             bc, 
                                             average_slice, 
                                             )

        new_rots = rots - ck_deltak
        bc = bind_weights(params, new_rots, qc)
        cost_minus, _, _ = cost_func_evaluate(noise,
                                              shots,
                                              cost_fn, 
                                              bc, 
                                              average_slice, 
                                              )

        delta = cost_plus - cost_minus
        gradient_array = delta / (2 * ck_deltak)
        #need to return an array to match parameter shift
    else:
        raise Exception(f'Gradient type {gradient_type} is not an allowed choice')
    return gradient_array   
    
def define_parameters(qubits:int,                  
                      mode:int, 
                      num_params:int) -> list:
    """Set up parameters and initialise text
    
    Parameters
    ----------
    qubits: int - The number of qubits in the circuit
    mode: int - Controls setting the circuit up in different modes

    Returns
    -------
    params: list
        A list of parameters (the texts)

    """
    params = []
    if mode in [1, 2, 3, 4, 6, ]:
        for i in range(num_params):
            text = "param " + str(i)
            params.append(FreeParameter(text))
        return params
    else:   
        raise Exception(f'Mode {mode} has not been coded for')
    
def vqc_circuit(qubits: int,
                mode:int,
                noise:bool,
                layers:int,
                params:list) -> Circuit:
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

    qc = Circuit()
    if mode == 1:
        for layer in range(layers):
            offset = layer * qubits * 2
            for i in range(qubits):
                qc.h(i)
                qc.ry(i, params[i+offset],)
                qc.rx(i, params[qubits+i+offset]),
            for i in range(qubits):
                if i < qubits-1:
                    qc.cnot(i,i+1)
                else:
                    qc.cnot(i,0)
    elif mode == 2:
        for layer in range(layers):
            offset = layer * qubits * 2
            for i in range(qubits):
                qc.rx(i, params[i+offset],)
            for i in range(qubits):
                if i < qubits-1:
                    qc.xx(i, i+1,params[qubits+i+offset],)
                else:
                    qc.xx(i, 0,params[qubits+i+offset],)
    elif mode == 3:
        for layer in range(layers):
            offset = layer * qubits * 2
            for i in range(qubits):
                qc.h(i)
                if i < qubits-1:
                    qc.zz(i, i+1, params[qubits+i+offset],)
                else:
                    qc.zz(i, 0, params[qubits+i+offset],)
            for i in range(qubits):
                qc.rz(i, params[i+offset],)
                qc.h(i)
    elif mode == 4:
        for layer in range(layers):
            offset = layer * qubits
            for i in range(qubits):
                qc.rx(i, params[i+offset],)
    elif mode == 5:
    #test mode
        if qubits != 5:
            raise Exception(f'test mode {mode} is only to be used with 5 qubits.  {qubits} qubits are specified')
        qc.x(1)
        qc.x(3)
        qc.x(4)

    elif mode == 6:
        for layer in range(layers):
            offset = layer * qubits * 2
            for i in range(qubits):
                qc.h(i)
                qc.ry(i, params[i+offset],)
                qc.rx(i, params[qubits+i+offset],)
    else:
        raise Exception(f'Mode {mode} has not been coded for')
        
    qc.measure(range(qubits))

    return qc

def create_initial_rotations(qubits: int,
                             mode: int,
                             layers:int,
                             hot_start:bool=False,
                             bin_hot_start_list: list=False,)-> np.ndarray: 
    """Initialise parameters with random weights

    Parameters
    ----------
    qubits : int
        The number of qubits in the circuit
    mode : int
        Controls setting the circuit up in different modes
    layers : int
        The number of layers
    hot_start : bool
        If true hot start values are used
    bin_hot_start_list : list
        Binary list containing the hot start values

    Returns
    -------
    init_rots: array
        initial rotations
    
    """
    
    if mode in [1, 2, 3, 6,]:
        param_num = 2 * qubits * layers
    elif mode == 4:
        param_num = qubits * layers
    else:
        raise Exception(f'Mode {mode} is not yet coded')
    if hot_start:
        if layers in [1]:
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

def bind_weights(params:list, rots:list, qc:Circuit) -> Circuit:
    """Bind parameters to rotations and return a bound quantum circuit

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
        param_name = str(params[i])
        binding_dict[param_name] = rot
    bc = qc.make_bound_circuit(binding_dict)
    return(bc)

def detect_quantum_GPU_support()-> bool:
    """Detect if a GPU is available for quantum simulations"""
    return False