from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter

from modules.helper_functions_tsp import (vqc_circuit, 
                                          cost_func_evaluate, 
                                          my_gradient, 
                                          cost_fn_fact
                                          )

import numpy as np
import pytest as py

from pathlib import Path
from modules.config import NETWORK_DIR

def my_cost_function1(bit_string_list:list) -> int:
    """A simple cost function for testing"""
    if bit_string_list == [0]:
        return 0 
    elif bit_string_list == [1]:
        return 1
    else:
        raise Exception('Invalid bit string list')

def test_gradient_1():
    """Test a circuit with one parameter"""
    a = Parameter('a')
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.rx(a, q[0])
    qc.measure_all()
    init_rots = [0]
    params = [a]
    actual_results = my_gradient(my_cost_function1, 
                                 False,
                                 qc, 
                                 params, 
                                 init_rots
                                 )
    print(actual_results)
    expected_results = np.array([0])
    assert actual_results == py.approx(expected_results, abs=0.1)

def test_gradient_2():
    """Test a circuit with one parameter"""
    a = Parameter('a')
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.rx(a, q[0])
    qc.measure_all()
    init_rots = [np.pi]
    params = [a]
    actual_results = my_gradient(my_cost_function1, 
                                 False,
                                 qc, 
                                 params, 
                                 init_rots
                                 )
    expected_results = np.array([0])
    assert actual_results == py.approx(expected_results, abs=0.1)

def test_gradient_3():
    """Test a circuit with one parameter"""
    a = Parameter('a')
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.rx(a, q[0])
    qc.measure_all()
    init_rots = [np.pi/2]
    params = [a]
    actual_results = my_gradient(my_cost_function1, 
                                 False,
                                 qc, 
                                 params, 
                                 init_rots
                                 )
    expected_results = np.array([0.5])
    assert actual_results == py.approx(expected_results, abs=0.1)

def test_gradient_4():
    """Test a circuit with two parameters and compare to qiskit results"""
    a = Parameter('a')
    b = Parameter('b')
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.h(q)
    qc.rz(a, q[0])
    qc.rx(b, q[0])
    qc.measure_all() 
    init_rots = [np.pi / 4, np.pi / 2]
    params = [a, b]
    actual_results = my_gradient(my_cost_function1, 
                                 False,
                                 qc, 
                                 params, 
                                 init_rots
                                 )
    expected_results = np.array([-0.353, 0.0]) #qiskit results
    assert actual_results == py.approx(expected_results, abs=0.1)

def test_simple_circuit():
    """test a simple circuit with known output"""
    qubits = 5
    params = []
    #mode = 3
    mode = 5
    locations = 5
    file = 'five_d.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    distance_array = np.genfromtxt(filename)
    gray = True
    shots = 1024
    cost_fn = cost_fn_fact(locations,
                           distance_array, 
                           gray
                           )
    qc = vqc_circuit(qubits, 
                     params, 
                     mode,
                     )

    actual_result, _ , _ = cost_func_evaluate(cost_fn, 
                                              False,
                                              qc, 
                                              shots
                                              )
    expected_result = 21.0
    assert actual_result == expected_result