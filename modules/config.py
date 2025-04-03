import numpy as np
# General control data
CONTROL_DIR = 'control'
CONTROL_FILE = 'control_parameters.csv'

NETWORK_DIR = 'networks'
DATA_SOURCES = 'data_sources.json'
GRAPH_DIR = 'graphs'
RESULTS_DIR = 'results'
RESULTS_FILE = 'results.csv'

ENCODING = 'utf-8-sig'              # Encoding of csv file

VERBOSE = False                     # controls how much is printed
CACHE_MAX_SIZE = 500_000            # maximum size of the cache.

# configuration information used in manual runs

LOCATIONS = 5                       # number of locations to be visited
SLICES = [1, 0.75, 0.5, 0.25]                    
                                    # Slices to use when calculating the gradient
                                    #[1, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.05] 
                                    # For example, 0.2 means that the best 20% 
                                    # of distances found is included in the average.
SHOTS = 1024                        # shots used for each call of the quantum circuit
ITERATIONS =  20                    # updates, or iterations
GRAY = True                         # Use Gray codes
HOT_START = True                    # Make a hot start

GRADIENT_TYPE = 'SPSA'              # controls the optimiser used
                                    # quantum - 'parameter_shift' - default
                                    # quantum - 'SPSA' is a stochastic gradient descent
                                    # ml - 'SGD' stochastical
                                    # ml - 'adam' 

DECODING_FORMULATION = 'original'   # 'original' or 'new' - new is formulation from paper

#information needed in quantum manual runs
MODE = 2                            # MODE = 1 - rxgate, rygate, cnot gates
                                    # MODE = 2 - rxgate, XX gates -can be used with Hot Start
ALPHA = 0.602                       # constant that controls the learning rate for SPSA decays
BIG_A = 50                          # A for SPSA
C = np.pi/10                        # initial CK for SPSA
ETA = 0.01                          # eta - learning rate for parameter shift
GAMMA = 0.101                       # constant that determines how quickly the SPSA perturbation decays
S = 0.5                             # parameter for parameter shift.  Default is 0.5                                   
PRINT_FREQUENCY = 5                 # how often results are printed out
CHANGE_EACH_PARAMETER = True        # Iterate through each parameter in the circuit
PLOT_PARAMETER_EVALUATION = True    # Plot the evaluation of each parameter         
ROTATIONS = 10                      # number of rotations sampled in parameter graphs

#information needed in ML manual runs
NUM_LAYERS = 2                      #number of layers in the mode
STD_DEV = 0.5                       #standard deviation for weight randomization
LR = 0.0001                         #Learning rate
MOMENTUM = 0.000                    #momentum for optimizer
WEIGHT_DECAY = 0.0002               #importance of L2 regularization in optimiser
#OPTIMIZER = 'SGD'                   #optimizer to use
#                                    #options: 'Adam', 'SGD', 'RMSprop'
