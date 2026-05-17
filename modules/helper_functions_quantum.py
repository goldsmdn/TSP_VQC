#helper functions for quantum circuit construction and evaluation
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
    sdk = find_sdk(target)
    binding_dict = {}
    for i, rot in enumerate(rots):
        param_name = str(params[i])
        match sdk:
            case 'aws':
                binding_dict[param_name] = rot
            case 'qiskit':
                binding_dict[param_name] = rot
            case _:
                raise Exception(f'SDK {sdk} has not been coded for')
    match sdk:
        case 'aws':
            bc = qc.make_bound_circuit(binding_dict)
        case 'qiskit':
            bc = qc.assign_parameters(binding_dict)
        case _:
            raise Exception(f'SDK {sdk} has not been coded for')
    return(bc)

def define_parameters(            
        mode:int, 
        num_params:int) -> list:
    """Set up parameters and initialise text
    
    Parameters
    ----------
    #qubits: int - The number of qubits in the circuit
    mode: int - Controls setting the circuit up in different modes

    Returns
    -------
    params: list
        A list of parameters (the texts)

    """
    params = []
    if mode in [1, 2, 3, 4, 6, 7, 12, 13, ]:
        for i in range(num_params):
            text = "param_" + str(i)
            params.append(FreeParameter(text))
        return params
    else:   
        raise Exception(f'Mode {mode} has not been coded for')
    

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

    circuit_sdk = MODE_DISPATCH[mode]['sdk']
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