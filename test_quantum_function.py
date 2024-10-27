from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from quantum_functions import bind_weights, my_gradient
import numpy as np
import pytest as py

def my_cost_function1(bit_string_list):
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
    actual_results = my_gradient(my_cost_function1, qc, params, init_rots, np.pi/2)
    expected_results = [0]

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
    actual_results = my_gradient(my_cost_function1, qc, params, init_rots, np.pi/2)
    expected_results = [0]

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
    actual_results = my_gradient(my_cost_function1, qc, params, init_rots, np.pi/2)
    expected_results = [0.5]

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
    actual_results = my_gradient(my_cost_function1, qc, params, init_rots, np.pi/2)
    expected_results = [-0.353, 0.0] #qiskit results

    assert actual_results == py.approx(expected_results, abs=0.1)



