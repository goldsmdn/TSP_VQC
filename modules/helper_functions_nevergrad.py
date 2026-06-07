#nevergrad helper functions
import numpy as np

from typing import Callable             # for function docs.
from braket.circuits import Circuit     # for function docs.
from qiskit.circuit import Parameter
from modules.helper_functions_tsp import (
        bind_weights,
        cost_func_evaluate,
        )


def validate_sigma_list(sigma):
    for items in sigma:
        if items <= 0:
            raise ValueError(f'{sigma=} should only contain positive values, {items} found.')
        
def ng_cost_function_fact(
    qc:Circuit,
    target:str,
    noise_bool:bool,
    shots:int,
    cost_fn:Callable,
    mps:bool,
    params:Parameter,
    init_rots:np.array,
    )->float:
    """returns a cost function for Nevergrad depending only on rotations"""
    def ng_cost_function(delta:np.array)-> float:

        rots = init_rots + delta
        rots = np.mod(rots, 2*np.pi)   # handle periodicity

        bc = bind_weights(
            params=params, 
            rots=rots, 
            qc=qc,
            target=target,
        )
        cost, lowest, _ = cost_func_evaluate(
            noise_bool=noise_bool,
            shots=shots,
            cost_fn=cost_fn,
            model=bc,
            target=target,
            mps=mps,
            average_slice=1,
        )    
        return cost, lowest
    return ng_cost_function