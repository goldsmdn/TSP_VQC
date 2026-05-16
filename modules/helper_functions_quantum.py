#helper functions for quantum circuit construction and evaluation
from braket.circuits import Circuit
from braket.parametric import FreeParameter

def find_mode(target:str) -> str:
    """Find the mode (braket or qiskit) for a given target

    Parameters
    ----------
    target: str
        The target to find the mode for.  This is a key in the TARGETS dictionary in config.py

    Returns
    -------
    mode: str
        The mode for the given target, either 'braket' or 'qiskit' that determines subsequent processing.
    """
    from modules.config import TARGETS
    return TARGETS[target]['mode']

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
        The target to find the mode for.  This is a key in the TARGETS dictionary

    Returns
    -------
    bc: Quantum Circuit
        A quantum circuit with including bound weights, ready to run an evaluation
    """
    mode = find_mode(target)
    binding_dict = {}
    for i, rot in enumerate(rots):
        param_name = str(params[i])
        match mode:
            case 'aws':
                binding_dict[param_name] = rot
            case 'qiskit':
                binding_dict[param_name] = rot
    match mode:
        case 'aws':
            bc = qc.make_bound_circuit(binding_dict)
        case 'qiskit':
            bc = qc.assign_parameters(binding_dict)
    return(bc)