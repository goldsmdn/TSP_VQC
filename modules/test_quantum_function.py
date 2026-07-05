from pathlib import Path

import numpy as np
import pytest as py
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister

from classes.MyDataLogger import MyDataLogger, MySubDataLogger
from modules.config import NETWORK_DIR
from modules.helper_functions_tsp import (
    cost_fn_fact,
    cost_func_evaluate,
    my_gradient,
    vqc_circuit,
)
from modules.quantum_circuits import find_maximum_qubits_needed


def my_cost_function1(bit_string_list: list) -> int:
    """A simple cost function for testing"""
    if bit_string_list == [0]:
        return 0
    elif bit_string_list == [1]:
        return 1
    else:
        raise Exception('Invalid bit string list')


def test_gradient_1_():
    """Test a circuit with one parameter_qiskit"""
    s = 0.5
    shots = 1024
    target = 'local_qiskit'
    mps = False
    gradient_type = 'parameter_shift'
    a = Parameter('a')
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.rx(a, q[0])
    qc.measure_all()
    init_rots = [0]
    params = [a]
    actual_results = my_gradient(
        qubits=1,
        noise_bool=False,
        shots=shots,
        s=s,
        gradient_type=gradient_type,
        cost_fn=my_cost_function1,
        qc=qc,
        params=params,
        rots=init_rots,
        average_slice=1,
        target=target,
        mps=mps,
    )
    expected_results = np.array([0])
    assert actual_results == py.approx(expected_results, abs=0.1)


def test_gradient_2():
    """Test a circuit with one parameter"""
    noise_bool = False
    shots = 1024
    s = 0.5
    gradient_type = 'parameter_shift'
    target = 'local_qiskit'
    mps = False
    a = Parameter('a')
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.rx(a, q[0])
    qc.measure_all()
    init_rots = [np.pi]
    params = [a]
    actual_results = my_gradient(
        qubits=1,
        noise_bool=noise_bool,
        shots=shots,
        s=s,
        gradient_type=gradient_type,
        cost_fn=my_cost_function1,
        qc=qc,
        params=params,
        rots=init_rots,
        average_slice=1,
        target=target,
        mps=mps,
    )
    expected_results = np.array([0])
    assert actual_results == py.approx(expected_results, abs=0.1)


def test_gradient_3():
    """Test a circuit with one parameter"""
    noise_bool = False
    shots = 1024
    s = 0.5
    gradient_type = 'parameter_shift'
    target = 'local_qiskit'
    mps = False
    a = Parameter('a')
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.rx(a, q[0])
    qc.measure_all()
    init_rots = [np.pi / 2]
    params = [a]
    actual_results = my_gradient(
        qubits=1,
        noise_bool=noise_bool,
        shots=shots,
        s=s,
        gradient_type=gradient_type,
        cost_fn=my_cost_function1,
        qc=qc,
        params=params,
        rots=init_rots,
        average_slice=1,
        target=target,
        mps=mps,
    )
    expected_results = np.array([0.5])
    assert actual_results == py.approx(expected_results, abs=0.1)


def test_gradient_4():
    """Test a circuit with two parameters and compare to qiskit results"""
    noise_bool = False
    shots = 1024
    s = 0.5
    gradient_type = 'parameter_shift'
    target = 'local_qiskit'
    mps = False
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
    actual_results = my_gradient(
        qubits=1,
        noise_bool=noise_bool,
        shots=shots,
        s=s,
        gradient_type=gradient_type,
        cost_fn=my_cost_function1,
        qc=qc,
        params=params,
        rots=init_rots,
        average_slice=1,
        target=target,
        mps=mps,
    )
    expected_results = np.array([-0.353, 0.0])  # qiskit results
    assert actual_results == py.approx(expected_results, abs=0.1)


def test_simple_circuit():
    """Test a simple circuit with known output"""
    shots = 1024
    qubits = 5
    params = []
    mode = 5
    locations = 5
    gray = True
    noise_bool = False
    formulation = 'original'
    target = 'local_qiskit'
    file = 'five_d.txt'
    filename = Path(NETWORK_DIR).joinpath(file)
    distance_array = np.genfromtxt(filename)
    cost_fn = cost_fn_fact(
        locations=locations,
        # qubits=qubits,
        gray=gray,
        formulation=formulation,
        distance_array=distance_array,
        # target=target,
    )

    qc = vqc_circuit(
        qubits=qubits,
        mode=mode,
        noise_bool=False,
        layers=1,
        params=params,
        target=target,
    )

    actual_result, _, _ = cost_func_evaluate(
        qubits=qubits,
        noise_bool=noise_bool,
        shots=shots,
        cost_fn=cost_fn,
        model=qc,
        target=target,
        mps=False,
        average_slice=1,
    )
    expected_result = 21.0
    assert actual_result == expected_result


def test_calculate_parameter_numbers_1():
    """Test parameter numbers with input 2,1"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid=datalogger.runid)
    sdl.qubits = 3
    sdl.mode = 1
    sdl.layers = 2
    sdl.target = 'local_qiskit_test'
    expected_result = 16
    actual_result = sdl.calculate_parameter_numbers()
    assert expected_result == actual_result


def test_calculate_parameter_numbers_2():
    """Test parameter numbers with input 2,1 for standard"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid=datalogger.runid)
    sdl.qubits = 3
    sdl.mode = 1
    sdl.layers = 2
    sdl.target = 'local_qiskit'
    expected_result = 12
    actual_result = sdl.calculate_parameter_numbers()
    assert expected_result == actual_result


def test_find_maximum_qubits_needed():
    input_dict = {0: 0, 1: 1, 2: 10, 3: 9}
    actual_output = find_maximum_qubits_needed(input_dict)
    expected_output = 11
    assert actual_output == expected_output
