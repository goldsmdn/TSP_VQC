#helper functions for quantum circuit construction and evaluation
import math
import random

from braket.circuits import Circuit
from braket.parametric import FreeParameter
from qiskit.circuit import Parameter
from qiskit import transpile

from qiskit_ibm_runtime.fake_provider import FakeAuckland

from modules.helper_functions_general import (
    find_logical_to_physical_dictionary, 
    find_qubits_measured, 
    find_valid_device_loop
)

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

from modules.config import MODE_DISPATCH

import numpy as np

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

def bind_weights(params:list, 
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
    
def vqc_circuit(qubits: int,
                mode:int,
                noise:bool,
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

    #circuit_sdk = MODE_DISPATCH[mode]['sdk']
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

    if noise:
        backend = FakeAuckland()
        qc = transpile(qc, backend=backend)
    return qc

def define_parameters(                
        mode:int, 
        num_params:int
        ) -> list:
    """Set up parameters and initialise text
    
    Parameters
    ----------
        mode: int - Controls setting the circuit up in different modes

    Returns
    -------
    params: list
        A list of parameters (the texts)

    """
    params = []
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
    return params

def create_initial_rotations(qubits: int,
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
        #if layers in [1]:
        #    raise Exception('Cannot use a hot start for mode {mode}')
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