from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2
import random
import math
import copy
from helper_functions_tsp import find_stats
import numpy as np

def cost_func_evaluate(cost_fn, bc: QuantumCircuit, 
                       shots: int = 1024) -> float:
    """evaluate cost function"""
    sampler = SamplerV2()
    job = sampler.run([bc])
    results = job.result()
    counts = results[0].data.meas.get_counts()
    cost, lowest, lowest_energy_bit_string = find_stats(cost_fn, counts, shots)
    return(cost, lowest, lowest_energy_bit_string)


def my_gradient(cost_fn, qc: QuantumCircuit, 
                params: list, rots: list, epsilon: float = np.pi/2, 
                shots:  int=1024, verbose: bool=False) -> list:
    """calculate gradient for a quantum circuit with parameters and rotations"""
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
    """set up parameters and initialise text"""
    params = []
    if mode in [1,2]:
        for i in range(qubits):
            #text1 = "param " + str(i)
            #text2 = "param " + str(2*i +1 )
            text1 = "param " + str(i)
            text2 = "param " + str(qubits+i)
            params.append(Parameter(text1))
            params.append(Parameter(text2))
        return params
    else:   
        raise Exception(f'Mode {mode} has not been coded for')

def vqc_circuit(qubits: int, params: list, mode:int=1) -> QuantumCircuit:
    """set up a variational quantum circuit"""
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

def create_initial_rotations(qubits: int) -> list:
    """initialise parameters with random weights"""
    init_rots= [random.random() * 2 * math.pi for i in range(qubits * 2)]
    return(init_rots)

def bind_weights(params:list, rots:list, qc:QuantumCircuit) -> QuantumCircuit:
    """bind parameters to rotations and return a bound quantum circuit"""
    binding_dict = {}
    for i, rot in enumerate(rots):
        binding_dict[str(params[i])] = rot
    bc = qc.assign_parameters(binding_dict)
    return(bc)