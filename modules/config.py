import numpy as np
# General control data - directories, file names, etc.
CONTROL_DIR = 'control'
CONTROL_FILE = 'control_parameters.csv'
NETWORK_DIR = 'networks'
DATA_SOURCES = 'data_sources.json'
GRAPH_DIR = 'graphs'
RESULTS_DIR = 'results'
RESULTS_FILE = 'results.csv'
ENCODING = 'utf-8-sig'              # Encoding of csv file

#General control parameters - verbosity, cache size, etc.
VERBOSE = False                     # controls how much is printed
CACHE_MAX_SIZE = 50_000_000         # maximum size of the cache.
PLOT_TITLE = False                  # Plot titles with graphs.  Not needed for publication.

# configuration information used in ALL manual runs

LOCATIONS = 4                       # number of locations to be visited          
                                    # Slices to use when calculating the gradient
                                    #[1, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.05] 
                                    # For example, 0.2 means that the best 20% 
                                    # of distances found is included in the average.
SHOTS = 1_024                       # shots used for each call of the quantum circuit

ITERATIONS =  10                    # updates, or iterations
PRINT_FREQUENCY = 15                # how often results are printed out
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
MODE = 6                            # MODE = 1 - rxgate, rygate, cnot gates
                                    # MODE = 2 - rxgate, XX gates -can be used with Hot Start
                                    # MODE = 3 - IQP based
                                    # MODE = 4 - rxgate
                                    # MODE = 5 - test mode
                                    # MODE = 6 - rxgate, ry gate
                                    # MODE = 8 - input is all zeros - with sine activation
                                    # MODE = 9 - input is 0.5 - with sine activation
                                    # MODE = 18 - input is all zeros - with sigmoid activation
                                    # MODE = 19 - input is 0.5 - with sigmoid activation
SLICES = [0.8]                      # Slices to use when calculating the gradient                                   
ALPHA = 0.602                       # constant that controls the learning rate for SPSA decays
BIG_A = 25                          # A for SPSA
C = np.pi/10                        # initial CK for SPSA
ETA = 0.1                           # eta - learning rate for parameter shift
GAMMA = 0.101                       # constant that determines how quickly the SPSA perturbation decays
S = 0.5                             # parameter for parameter shift.  Default is 0.5 
SIMULATE_NOISE = False              # Simulate noise in the quantum circuit

ROTATIONS = 10                      # number of rotations sampled in parameter graphs

#information needed in ML manual runs:
STD_DEV = 0.05                      #standard deviation for warm start weight randomization
LR = 1e-3                           #Learning rate
MOMENTUM = 0.9                     
WEIGHT_DECAY = 0.0006               #importance of L2 regularization in optimiser
                                    #options: 'Adam', 'SGD', 'RMSprop'                                 
