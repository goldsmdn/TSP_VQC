#quantum circuits.py

from braket.circuits import Circuit
from braket.parametric import FreeParameter
import numpy as np

from qiskit import QuantumCircuit


def mode_1(context_dict:dict) -> QuantumCircuit:
    """Creates a simple circuit with multiple layera of Hadamards, 
        RY and RX rotations and one layer of CNOTs

    Parameters
    ----------
    context_dict: dict
        A dictionary containing the context for the circuit, including the number of qubits, parameters, and layers

    Returns
    -------
    qc: Quantum Circuit
        A quantum circuit 
    """

    qubits = context_dict['qubits']
    params = context_dict['params'] 
    layers = context_dict['layers']

    qc = QuantumCircuit(qubits)
    for layer in range(layers):
        offset = layer * qubits * 2
        for i in range(qubits):
            qc.h(i)
            qc.ry(params[i+offset], i)
            qc.rx(params[qubits+i+offset], i)
        for i in range(qubits):
            if i < qubits-1:
                qc.cx(i,i+1)
            else:
                qc.cx(i,0)
    return qc

def mode_13(context_dict:dict) -> Circuit:
    """An IPQ inspired circuit 

    Parameters
    ----------
    context_dict: dict
        A dictionary containing the context for the circuit, including the 
        number of qubits, parameters, and layers

    Returns
    -------
    qc: Quantum Circuit
        A quantum circuit 
    """

    qubits = context_dict['qubits']
    params = context_dict['params'] 
    layers = context_dict['layers']
    qubit_dict = context_dict['qubit_dict']
    qubits_measured = context_dict['qubits_measured']

    inner = Circuit()
    for layer in range(layers):
        print(f'{qubit_dict=}, {qubits_measured=}, {qubits=}')
        offset = layer * qubits_measured * 2
        for i in range(qubits_measured):
            #hadamard
            inner.rz(qubit_dict[i], np.pi/2)
            inner.rx(qubit_dict[i], np.pi/2)
            inner.rz(qubit_dict[i], np.pi/2)
            inner.rz(qubit_dict[i], params[i+offset])
            if i < qubits_measured-1:
                inner.cz(qubit_dict[i], qubit_dict[i+1])
            else:
                inner.cz(qubit_dict[i], qubit_dict[0])
        for i in range(qubits_measured):
            inner.rz(qubit_dict[i], params[qubits_measured+i+offset],)
            # hadamdard
            inner.rz(qubit_dict[i], np.pi/2)
            inner.rx(qubit_dict[i], np.pi/2)
            inner.rz(qubit_dict[i], np.pi/2)
        print(f'After circuit set up, the verbatim box receives the following circuit{inner.qubits}')
        print(f'The qubit dictionary is {qubit_dict}')
        qc = Circuit().add_verbatim_box(inner)  
    return qc

    # find which qubits are measured and the logical to physical dictionary for the device
