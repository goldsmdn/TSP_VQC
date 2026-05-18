from collections.abc import Set

import numpy as np
#from qiskit import circuit
#from torch import device

from modules.quantum_circuits import (
    mode_1,
    mode_2,
    mode_3,
    mode_4,
    mode_5, 
    mode_6,
    mode_7,
    mode_13,
)

# General control data - directories, file names, etc.

CONTROL_DIR = 'control'
CONTROL_FILE = 'control_parameters.csv'
NETWORK_DIR = 'networks'
DATA_SOURCES = 'data_sources.json'
GRAPH_DIR = 'graphs'
RESULTS_DIR = 'results'
RESULTS_FILE = 'results.csv'
ENCODING = 'utf-8-sig'              # Encoding of csv file
AWS = True                          # Whether runs are on AWS or Qiskit.

CEPHUS_DEVICE = 'arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q'

TARGET = 'local_aws'                   # Options from TARGETS dictionary below.  This controls which 
                                # quantum device is used and whether the emulator is used.

TARGETS = {
    'local_aws': { #test on local AWS simulator
        'type': 'local_aws',
        'emulator': True,
        'sdk': 'aws',
    },
    'local_qiskit': { #test on local qiskit simulator
        'type': 'local_qiskit',
        'emulator': True,
        'sdk': 'qiskit',
    },
    'local_aws_test': { #test on local aws simulator using Cephus connectivity
        'type': 'local_aws',
        'emulator': True,
        'sdk': 'aws',
    },
    'cephus': { #production run on Rigetti Cephus
        'type': 'aws',
        'arn': CEPHUS_DEVICE,
        'emulator': False,
        'sdk': 'aws',
    },
    'cephus_em': {#emulator run on Rigetti Cephus - not currently provided by Rigetti, but could be added in the future
        'type': 'aws',
        'arn': CEPHUS_DEVICE,
        'emulator': True,
    },
    'ml': { #ML model - not a quantum device
        'type': 'ml',
        'emulator': False,
        'sdk': 'ml',
    },
}

#General control parameters - verbosity, cache size, etc.
VERBOSE = False                     # controls how much is printed
CACHE_MAX_SIZE = 50_000_000         # maximum size of the cache.
PLOT_TITLE = False                  # Plot titles with graphs.  Not needed for publication.

# configuration information used in ALL manual runs

LOCATIONS = 4                       # number of locations to be visited          
SHOTS = 1_024                       # shots used for each call of the quantum circuit

ITERATIONS =  20                    # updates, or iterations
PRINT_FREQUENCY = 5                 # how often results are printed out
GRAY = False                        # Use Gray codes
HOT_START = False                   # Make a hot start
GRADIENT_TYPE = 'SPSA'              # controls the optimiser used
                                    # quantum - 'parameter_shift' - default
                                    # quantum - 'SPSA' is a stochastic gradient descent
                                    # ml - 'SGD' stochastical
                                    # ml - 'SGD+X' stochastical with Xavier initialization
                                    # ml - 'Adam' 
DECODING_FORMULATION = 'original'   # 'original' or 'new' - new is formulation from paper
NUM_LAYERS = 1                      # number of layers in the model

#information needed in QML manual runs:
MODE = 13                           # See list of allowed modes in MODE_DISPATCH below.  
#This controls the structure of the variational quantum circuit used in the QML runs.  
#The modes are described in the function that sets up the variational quantum circuit 
#in helper_functions_quantum.py.  
#The mode also controls which SDK is used - AWS, Qiskit or ML.  

MODE_DISPATCH = {
    1: {'circuit':mode_1, #Qiskit rxgate, rygate, cnot gates
        'sdk': 'qiskit',
        'params_per_qubit': 2,
        'hot_start_valid': False},
    2: {'circuit':mode_2, #Qiskit rxgate, XX gates -can be used with Hot Start
        'sdk': 'qiskit',
        'params_per_qubit': 2,
        'hot_start_valid': True},
    3: {'circuit':mode_3, #Qiskit IQP based
        'sdk': 'qiskit',
        'params_per_qubit': 2,
        'hot_start_valid': False},
    4: {'circuit':mode_4, #Qiskit rxgate
        'sdk': 'qiskit',
        'params_per_qubit': 1,
        'hot_start_valid': True},
    5: {'circuit':mode_5, #Qiskit test mode
        'sdk': 'qiskit',
        'params_per_qubit': 2,
        'hot_start_valid': False},
    6: {'circuit':mode_6, #Qiskit rxgate, ry gate
        'sdk': 'qiskit',
        'params_per_qubit': 2,
        'hot_start_valid': False},
    7: {'circuit':mode_7, #AWS rz gates, iswap gates
        'sdk': 'aws',
        'params_per_qubit': 2,
        'hot_start_valid': False},
    8: {'sdk': 'ml'}, #input is all zeros - with sine activation
    9: {'sdk': 'ml'}, #input is 0.5 - with sine activation
    13: {'circuit':mode_13,#AWS IQP with only RX, RZ and CZ
        'sdk': 'aws',
        'params_per_qubit': 2,
        'hot_start_valid': False},
    18: {'sdk': 'ml'}, #input is all zeros - with sigmoid activation    
    19: {'sdk': 'ml'}, #input is 0.5 - with sigmoid activation
}

SLICES = [0.8]                      # Slices to use when calculating the gradient  
                                    #[1, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.05] 
                                    # For example, 0.2 means that the best 20% 
                                    # of distances found is included in the average.                                 
ALPHA = 0.602                       # constant that controls the learning rate for SPSA decays
BIG_A = 25                          # A for SPSA
C = np.pi/10                        # initial CK for SPSA
ETA = 0.1                           # eta - learning rate for parameter shift
GAMMA = 0.101                       # constant that determines how quickly the SPSA perturbation decays
S = 0.5                             # parameter for parameter shift.  Default is 0.5 
SIMULATE_NOISE = False              # Simulate noise in the quantum circuit
MPS = False                         # Use MPS simulator

ROTATIONS = 10                      # number of rotations sampled in parameter graphs

#information needed in ML manual runs:
STD_DEV = 0.05                      #standard deviation for warm start weight randomization
LR = 1e-3                           #Learning rate
MOMENTUM = 0.9                     
WEIGHT_DECAY = 0.0006               #importance of L2 regularization in optimiser
                                    #options: 'Adam', 'SGD', 'RMSprop'    
                                    
VALID_QUBIT_LOOPS = {'ankaa':
                        {3: [  0, 1, 8, 7,], #convention - loops return to qubit 0 at the end and this is assumed in the code
                         8: [  0, 1, 2, 3, 10, 9,  8,  7,],
                         14: [ 0,  1,  2,  3,  4,  5,  6, 13, 12, 11, 10,  9, 8, 7,],
                         29: [ 0,  1,  2,  3,  4,  5,  6, 13, 12, 11, 10,  9, 8,
                              15, 16, 17, 18, 19, 20, 27, 26, 25, 24, 23, 22, 29, 28, 21,
                              14, 7,],
                         41:[ 0,  1,  2,  3,  4,  5,  6, 13, 12, 11, 10, 9, 8, 
                             15, 16, 17, 18, 19, 20, 27, 26, 25, 24, 23, 22, 29, 
                             30, 37, 44, 51, 58, 
                             57, 56, 49, 50, 43, 36, 35, 28, 21, 14,  7,],
                         49:[ 0,  1,  2,  3,  4,  5,  6, 13, 12, 11, 10, 9, 8, 
                             15, 16, 17, 18, 19, 20, 27, 26, 25, 24, 23, 22, 29, 
                             30, 37, 44, 51, 52, 53, 54, 55, 62, 61, 60, 59, 58, 
                             57, 56, 49, 50, 43, 36, 35, 28, 21, 14,  7,],
                             },
                    'cephus': 
                         {3: [  0, 1, 10, 9,], #convention - loops return to qubit 0 at the end and this is assumed in the code
                          8: [  0, 1, 2, 3, 12, 11, 10,  9,],
                          14: [ 0,  1,  2,  3,  4,  5,  6, 15, 14, 13, 12, 11, 10, 9,],
                          29: [ 0,  1,  2,  3,  4,  5,  6, 7, 16, 15, 14, 13, 12, 11, 10,
                              19, 20, 21, 22, 23, 24, 33, 32, 31, 30, 29, 28, 27,
                              18, 9,],
                          41:[ 0,  1,  2,  3,  4,  5,  6, 7, 16, 15, 14, 13, 12, 11, 10,
                              19, 20, 21, 22, 23, 24, 25, 26, 35, 34, 33, 32, 31, 30, 39, 48, 
                              49, 58, 57, 56, 55, 54, 
                              45, 36, 27, 18, 9,],
                          49:[ 0,  1,  2,  3,  4,  5,  6, 7, 16, 15, 14, 13, 12, 11, 10,
                              19, 20, 21, 22, 23, 24, 25, 26, 35, 34, 33, 32, 31, 30, 39, 48, 
                              49, 50, 51, 52, 61, 60, 59, 58, 57, 56, 65, 64, 55, 54, 
                              45, 36, 27, 18, 9,],
                             },
                     'local_aws_test':
                            {3: [  0, 1, 10, 9,],
                            14: [ 0,  1,  2,  3,  4,  5,  6, 15, 14, 13, 12, 11, 10, 9,],
                            }
                    }                         