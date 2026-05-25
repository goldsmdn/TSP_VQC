#quantum circuits.py
from IPython.display import display

from braket.circuits import Circuit
from braket.parametric import FreeParameter
import numpy as np

from qiskit import QuantumCircuit


LARGEST_QUANTUM_CIRCUIT_TO_PRINT = 15

def print_quantum_circuits(
        qubits:int,
        sdk_type:str,
        qc:QuantumCircuit,
        filename:str,
        ):
    """prints a quantum circuit using Qiskit or AWS formats"""
    if qubits < LARGEST_QUANTUM_CIRCUIT_TO_PRINT:
        match sdk_type:
            case 'aws':
                print(qc)
            case 'qiskit':
                fig = qc.draw("mpl", style="clifford",)
                display(fig)
                fig.savefig(filename, format='pdf', bbox_inches="tight")
            case '_':
                raise Exception(f'{sdk_type=} is not an allowed value')

def mode_1(context_dict:dict) -> QuantumCircuit:
    """Creates a simple Qiskit circuit with multiple layera of Hadamards, 
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

    qc = QuantumCircuit(qubits, qubits)
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

def mode_2(context_dict:dict) -> QuantumCircuit:
    """Creates a simple Qiskit circuit with multiple layers of Hadamards, 
        RX rotations and one layer of XX gates

    Parameters
    ----------
    context_dict: dict
        A dictionary containing the context for the circuit, 
        including the number of qubits, parameters, and layers

    Returns
    -------
    qc: Quantum Circuit
        A quantum circuit 
    """

    qubits = context_dict['qubits']
    params = context_dict['params'] 
    layers = context_dict['layers']

    qc = QuantumCircuit(qubits, qubits)
    for layer in range(layers):
        offset = layer * qubits * 2
        for i in range(qubits):
            qc.rx(params[i+offset], i)
        for i in range(qubits):
            if i < qubits-1:
                qc.rxx(params[qubits+i+offset], i, i+1,)
            else:
                qc.rxx(params[qubits+i+offset], i, 0,)
    return qc

def mode_3(context_dict:dict) -> QuantumCircuit:
    """Creates a simple Qiskit circuit inspired by IQP circuits, 
       with multiple layers of Hadamards, 
        RX and RZ rotations and one layer of CZ gates

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

    qc = QuantumCircuit(qubits, qubits)
    if layers > 1:
        raise Exception('Mode 3 is only coded for one layer')
    offset = layers * qubits * 2
    for i in range(qubits):
        qc.h(i)
        if i < qubits-1:
            qc.rzz(params[qubits+i+offset], i, i+1,)
        else:
            qc.rzz(params[qubits+i+offset], i, 0,)
    for i in range(qubits):
        qc.rz(params[i+offset], i)
        qc.h(i)
    return qc

def mode_4(context_dict:dict) -> QuantumCircuit:
    """Creates a simple Qiskit circuit with multiple layers of RX rotations

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

    qc = QuantumCircuit(qubits, qubits)
    for layer in range(layers):
        offset = layer * qubits * 2
        for i in range(qubits):
            qc.rx(params[i+offset], i)
    return qc

def mode_5(context_dict:dict) -> QuantumCircuit:
    """Creates a simple Qiskit circuit with multiple layers of Hadamards, 
        RX and RZ rotations and one layer of CNOT gates, but only 2 qubits

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
    #params = context_dict['params'] 
    #layers = context_dict['layers']

    if qubits != 5:
        raise Exception(f'test mode 5 is only to be used with 5 qubits.  {qubits} qubits are specified')
    
    qc = QuantumCircuit(qubits, qubits)
    qc.x(1)
    qc.x(3)
    qc.x(4)
    return qc

def mode_6(context_dict:dict) -> QuantumCircuit:
    """Creates a simple Qiskit circuit with multiple layers of Hadamards, 
        RX and RY rotations and one layer of CNOT gates

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

    qc = QuantumCircuit(qubits, qubits)
    for layer in range(layers):
        offset = layer * qubits * 2
        for i in range(qubits):
            qc.h(i)
            qc.ry(params[i+offset], i)
            qc.rx(params[qubits+i+offset], i)
    return qc

def mode_7(context_dict:dict) -> Circuit:
    """A simple circuit created in AWS Braket with multiple layers of RZ rotations and ISWAP gates

    Parameters
    ----------
    context_dict: dict
        A dictionary containing the context for the circuit, including the number of qubits, parameters, and layers

    Returns
    -------
    qc: Quantum Circuit
        A quantum circuit 
    """

    #qubits = context_dict['qubits']
    params = context_dict['params'] 
    layers = context_dict['layers']
    qubit_dict = context_dict['qubit_dict']
    qubits_measured = context_dict['qubits_measured']

    inner = Circuit()
    for layer in range(layers):
        offset = layer * qubits_measured * 2
        for i in range(qubits_measured):
            inner.rz(qubit_dict[i], params[i+offset])
            if i < qubits_measured-1:
                inner.iswap(qubit_dict[i],qubit_dict[i+1])
            else:
                inner.iswap(qubit_dict[i],qubit_dict[0])
            inner.rz(qubit_dict[i], params[qubits_measured+i+offset])
    print(f'After circuit set up, the verbatim box receives the following circuit{inner.qubits}')
    qc = Circuit().add_verbatim_box(inner)
    return qc

def mode_13(context_dict:dict) -> Circuit:
    """An IPQ inspired circuit created in AWS Braket

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
