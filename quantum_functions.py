from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2
import random
import math
import copy
from helper_functions_tsp import find_stats
import numpy as np

def cost_func_evaluate(cost_fn, bc: QuantumCircuit, 
                       shots: int = 1024) -> tuple:
    """evaluate cost function
    
    Parameters
    ----------
    cost_fn: function
        A function of a bit string evaluating a distance for that bit string
    bc: QuantumCircuit
        A quantum circuit with bound weights for which the energy is to be found
    shots: int
        The number of shots for which the quantum circuit is to be run

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
    cost, lowest, lowest_energy_bit_string = find_stats(cost_fn, counts, shots)
    return(cost, lowest, lowest_energy_bit_string)

def my_gradient(cost_fn, qc: QuantumCircuit, 
                params: list, rots: list, epsilon: float = np.pi/2, 
                shots:  int=1024, verbose: bool=False) -> list:
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
    epsilon: float
        Determines the rotation for evaluating the gradient
    shots: int
        The number of shots for which the quantum circuit is to be run in each estimation of a parameter point
    verbose: bool
        If True then more information is printed

    Returns
    -------
    gradient:list
        The gradient for each parameter
    """
    gradient = []
    
    new_rots = copy.deepcopy(rots)
    for i, rot in enumerate(rots):
        if verbose:
            print(f'processing {i}th weight')
            print(f'rot = {rot} i={i}')

        new_rots[i] = rot + epsilon
        if verbose:
            print(f'New rots+ = {new_rots}')
        bc = bind_weights(params, new_rots, qc)
        cost_plus, _, _ = cost_func_evaluate(cost_fn, bc, shots)

        new_rots[i] = rot - epsilon
        if verbose:
            print(f'New rots- = {new_rots}')
        bc = bind_weights(params, new_rots, qc)
        cost_minus, _, _ = cost_func_evaluate(cost_fn, bc, shots)

        delta = (cost_plus - cost_minus) / 2
        if verbose:
            print(f'cost+ = {cost_plus} cost- = {cost_minus}, delta = {delta}')
        gradient.append(delta)

    return gradient

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
        for i in range(qubits):
            text1 = "param " + str(i)
            text2 = "param " + str(qubits+i)
            params.append(Parameter(text1))
            params.append(Parameter(text2))
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
            qc.ry(params[2*i], i)
            qc.rx(params[2*i+1], i)
        for i in range(qubits):
            if i < qubits-1:
                qc.cx(i,i+1)
            else:
                qc.cx(i,0)
                #ensure circuit is fully entangled
    elif mode == 2:
        for i in range(qubits):
            qc.rx(params[2*i], i)
        for i in range(qubits):
                if i < qubits-1:
                    qc.rxx(params[2*i+1], i, i+1, )
                else:
                    qc.rxx(params[2*i+1], i, 0, )
                #ensure circuit is fully entangled
    else:
        raise Exception(f'Mode {mode} has not been coded for')
    qc.measure_all()
    return qc

def create_initial_rotations(qubits: int, mode: int, hot_start: bool=False) -> list:
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
    init_rots: list
        initial rotations
    
    """
    if mode in [1,2]:
        param_num = 2 * qubits
    else:
        raise Exception(f'Mode {mode} is not yet coded')
    if hot_start:
        init_rots = [0 for i in range(param_num)]
    else:
        init_rots= [random.random() * 2 * math.pi for i in range(param_num)]
    return(init_rots)

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