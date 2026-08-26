# configuration for graphs

import numpy as np

from modules.helper_functions_tsp import (
    find_nevergrad_optimizers,
)

# key fields

KEY_FIELDS = [  # default fields to be shown
    'quantum',
    'locations',
    'iteration_found',
    'best_dist_found',
    'best_dist',
    'quality',
    'error',
    'mode',
    'monte_carlo',
    'layers',
    'elapsed',
    'slice',
]

KEY_FIELDS_STATS = [  # used if only stats are required
    'locations',
    'iteration_found',
    'quality',
    'error',
    'shots',
    'elapsed',
    'cache_misses',
    'cache_hits',
]

KEY_FIELDS_ML_HYPER = [
    'locations',
    'iteration_found',
    'quality',
    'error',
    'shots',
    'elapsed',
    'lr',
    'weight_decay',
    'momentum',
]

KEY_FIELDS_ML_OPTIMIZER_COMPARISON = [
    'locations',
    'iteration_found',
    'quality',
    'error',
    'shots',
    'elapsed',
    'lr',
    'weight_decay',
    'gradient_type',
]


# filters

LOCATIONS = [10, 11, 12, 15, 17, 26, 42, 48]
LOCS_ANALYSED = 12  # used in Nevergrad
CACHE_LOCATIONS = [4, 5, 6, 7, 8, 9, 10, 11, 12]

FILTERS_LOCATIONS = {
    'locations': LOCATIONS,
}

FILTERS_CACHE_LOCATIONS = {
    'locations': CACHE_LOCATIONS,
}

FILTERS_GENERAL = {
    'formulation': 'original',
    'hot_start': False,
    'gray': False,
    'iterations': 250,
    'noise': False,
}

FILTERS_GENERAL_417 = {
    'formulation': 'original',
    'hot_start': False,
    'gray': False,
    'iterations': 417,
    'noise': False,
}

FILTERS_OVERALL = {
    'locations': LOCATIONS,
    'hot_start': False,
    'gray': False,
    'noise': False,
}

FILTERS_VQA_SPSA = {
    'quantum': True,
    'gradient_type': 'SPSA',
    'alpha': 0.602,
    'big_a': 25,
    'c': lambda c: np.isclose(c, np.pi / 10, atol=1e-3),
    'gamma': 0.101,
    'eta': 0.1,
    's': 0.5,
    'shots': 1_024,
}

FILTERS_VQA_SPSA2 = {
    'quantum': True,
    'shots': 1_024,
    'gradient_type': 'SPSA2',
    'iterations': 1_250,
    'layers': 1,
    'alpha': 0.602,
    'big_a': 100,
    'c': lambda c: np.isclose(c, np.pi / 10, atol=1e-3),
    'gamma': 0.101,
    'eta': 0.005,
    's': 0.5,
    # 'monte_carlo': False,
}

FILTERS_VQA_SPSA_417 = {  # 417 iterations - fair comparison
    'quantum': True,
    'slice': 1,
    'shots': 1_024,
    'gradient_type': 'SPSA',
    'iterations': 417,
    'layers': 1,
    'alpha': 0.602,
    'big_a': 25,
    'c': lambda c: np.isclose(c, np.pi / 10, atol=1e-3),
    'gamma': 0.101,
    'eta': 0.1,
    's': 0.5,
    # 'monte_carlo': False,
}

FILTERS_VQA_CMA = {
    'quantum': True,
    'shots': 1_024,
    'gradient_type': 'CMA',
    'iterations': 1_250,
    'sigma': 0.7,
    #    'monte_carlo': False,
}

FILTERS_VQA_CEPHEUS = {
    'quantum': True,
    'slice': 1,
    'shots': 1024,
    'mode': 21,
    'gradient_type': 'CMA',
    'iterations': lambda iter: iter > 4,
    'layers': 1,
    'target': 'cepheus',
    'locations': LOCATIONS,
    # 'monte_carlo': False,
    'hot_start': True,
}

FILTERS_SLICING = {
    'locations': [10, 11, 12, 15],
    'mode': 2,
    'mps': True,
    'monte_carlo': False,
}

FILTERS_MODEL_ANALYSIS = {
    'locations': 15,
    'slice': 1.0,
    'mps': True,
    # 'monte_carlo': False,
}

FILTERS_VQA_HOT_START = {
    'formulation': 'original',
    'gray': False,
    'iterations': 250,
    'mode': 2,
    'slice': lambda s: np.isclose(s, 1.0, atol=1e-3),
    'noise': False,
    # 'monte_carlo': False,
}

FILTERS_VQA_GRAY = {
    'formulation': 'original',
    'hot_start': False,
    'iterations': 250,
    'mode': 2,
    'slice': lambda s: np.isclose(s, 1.0, atol=1e-3),
    'noise': False,
    # 'monte_carlo': False,
}

FILTERS_VQA_FORMULATION = {
    'hot_start': False,
    'gray': False,
    'iterations': 250,
    'mode': 2,
    'slice': lambda s: np.isclose(s, 1.0, atol=1e-3),
    'noise': False,
    # 'monte_carlo': False,
}

FILTERS_VQA_NOISE = {
    'formulation': 'original',
    'hot_start': False,
    'gray': False,
    'iterations': 250,
    'slice': 0.8,
    'shots': 1_024,
    'mode': 2,
    #'monte_carlo': False,
}

FILTERS_VQA_NG1 = {
    'locations': 12,
    'shots': 1_024,
    'mode': 2,
    'gradient_type': list(find_nevergrad_optimizers()),
    # 'monte_carlo': False,
    # show all Nevergrad optmisers explored.
}


FILTERS_ML = {
    'quantum': False,
    'shots': 64,
    'std_dev': 0.05,
    'lr': lambda lr: np.isclose(lr, 2e-5, atol=1e-7),
    'weight_decay': 0.0006,
    'momentum': 0.8,
}

FILTERS_ML_LAYER_ANALYSIS = {'mode': 8, 'monte_carlo': False, 'hot_start': False}

FILTERS_ML_HOT_START_TRUE = {
    'formulation': 'original',
    'gray': False,
    'iterations': 250,
    'mode': 8,
    'layers': 1,
    'hot_start': True,
    'target': 'ml',
}

FILTERS_ML_HOT_START_FALSE = {
    'formulation': 'original',
    'gray': False,
    'iterations': 250,
    'mode': 8,
    'layers': 4,
}

FILTERS_ML_GRAY = {
    'formulation': 'original',
    'hot_start': False,
    'iterations': 250,
    'mode': 8,
    'slice': lambda s: np.isclose(s, 1.0, atol=1e-3),
    'layers': 4,
    # 'monte_carlo': False,
}

FILTERS_ML_FORMULATION = {
    'hot_start': False,
    'gray': False,
    'iterations': 250,
    'mode': 8,
    'layers': 4,
    'locations': lambda locs: locs < 26,
    # 'monte_carlo': False,
}

FILTERS_ML_MINIBATCH = {
    'quantum': False,
    'shots': lambda s: s != 2,
    'std_dev': 0.05,
    'lr': lambda lr: np.isclose(lr, 2e-5, atol=1e-7),
    'weight_decay': 0.0006,
    'momentum': 0.8,
    'mode': 8,
    'layers': 4,
    'gradient_type': 'SGD',
    'noise': False,
    # 'monte_carlo': False,
}

FILTERS_ML_OPTIMISERS_SGD = {
    'quantum': False,
    'shots': [64, 256],
    'std_dev': 0.05,
    'lr': lambda lr: np.isclose(lr, 2e-5, atol=1e-7),
    'weight_decay': 0.0006,
    'momentum': 0.8,
    'gradient_type': 'SGD',
    'layers': 4,
    # 'monte_carlo': False,
}

FILTERS_ML_OPTIMISERS_ADAM = {
    'quantum': False,
    'shots': [64, 256],
    'std_dev': 0.05,
    'lr': lambda s: np.isclose(s, 0.001, atol=1e-7),
    'weight_decay': 0.0032,
    'layers': 4,
    'gradient_type': 'Adam',
    # 'monte_carlo': False,
}

FILTERS_ML_ADAMS_HYPERPARAMETERS = {
    'quantum': False,
    'shots': 64,
    'std_dev': 0.05,
    'layers': 4,
    'locations': 12,
    'gradient_type': 'Adam',
    'noise': False,
}

FILTERS_ML_SGD_HYPERPARAMETERS = {
    'quantum': False,
    'shots': 64,
    'std_dev': 0.05,
    'momentum': lambda mom: np.isclose(mom, 0.8, atol=1e-7),
    'layers': 2,
    'locations': 10,
    'gradient_type': 'SGD',
    'mode': 8,
}

FILTERS_ML_INIT = {
    'gradient_type': 'SGD',
    'layers': 4,
    # 'monte_carlo': False,
}

FILTERS_MC_TRUE = {'monte_carlo': True}

FILTERS_MC_FALSE = {'monte_carlo': False}

FILTERS_QUANTUM_FALSE = {'quantum': False}

FILTERS_MODE_02 = {'mode': 2}

FILTERS_MODE_15 = {'mode': 15}

FILTERS_MODE_22 = {'mode': 22}

FILTERS_HOT_START = {
    'formulation': 'original',
    'gray': False,
    'noise': False,
    'mps': True,
    'hot_start': True,
}

FILTERS_COLD_START = {
    'formulation': 'original',
    'gray': False,
    'noise': False,
    'mps': True,
    'hot_start': False,
}
