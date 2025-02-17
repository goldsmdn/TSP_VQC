import numpy as np
#CONTROL_FILE = 'control/control_parameters.csv'
CONTROL_DIR = 'control'
CONTROL_FILE = 'control_parameters.csv'

NETWORK_DIR = 'networks'

ENCODING = 'utf-8-sig'              # Encoding of csv file

# configuration files loaded from control.csv in production runs
LOCATIONS = 4                       # number of locations to be visited
SLICES = [1, 0.75]                      
                                    # Slices to use when calculating the gradient
                                    #[1, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.05] 
                                    # For example, 0.2 means that the best 20% 
                                    # of distances found is included in the average.
SHOTS = 1024                        # shots used for each call of the quantum circuit
MODE = 2                            # MODE = 1 - rxgate, rygate, cnot gates
                                    # MODE = 2 - rxgate, XX gates -can be used with Hot Start
ITERATIONS = 50                     # updates, or iterations

GRAY = True                         # Use Gray codes
HOT_START = True                    # Make a hot start

GRADIENT_TYPE = 'SPSA'              # controls the optimiser used
                                    # 'parameter_shift' - default
                                    # 'SPSA' is a stochastic gradient descent

DECODING_FORMULATION = 'new'        # 'original' or 'new' - new is forumlation from paper

ALPHA = 0.602                       # constant that controls the learning rate for SPSA decays
BIG_A = 50                          # A for SPSA
C = np.pi/10                        # initial CK for SPSA
ETA = 0.02                          # eta - learning rate for parameter shift
GAMMA = 0.101                       # constant that determines how quickly the SPSA perturbation decays
S = 0.5                             # parameter for parameter shift.  Default is 0.5

# data sources
DATA_SOURCES = {  4 : {'file' : 'four_d.txt', 'best' : 21, 'source': 'QTAP'},
                  5 : {'file' : 'five_d.txt', 'best' : 19, 'source': 'TSPLIB'},
                  8 : {'file' : 'sim_dist_8_locs.txt', 'best' : 277.2, 'source': 'sim',
                     'points' : 'sim_points_8_locs.txt'},
                 11 : {'file' : 'dg11_d.txt', 'best' : 253, 'source': 'TSPLIB'},
                 15 : {'file' : 'p01_d.txt',  'best' : 291, 'source': 'TSPLIB'},
                 17 : {'file' : 'gr17_d.txt', 'best' : 2085, 'source': 'TSPLIB'},
                 26 : {'file' : 'fri26_d.txt', 'best' : 699, 'source': 'TSPLIB'},
                 42 : {'file' : 'dantzig42_d.txt', 'best' : 33_523, 'source': 'TSPLIB'},
                }

# configuration to control execution
VERBOSE = False                     # controls how much is printed
PRINT_FREQUENCY = 10                # how often results are printed out
CACHE_MAX_SIZE = 500_000             #maximum size of the cache.

CHANGE_EACH_PARAMETER = True        # Iterate through each parameter in the circuit
PLOT_PARAMETER_EVALUATION = True    # Plot the evaluation of each parameter         
READ_AND_WRITE_DATA = True          # Read and write data to csv file  

ROTATIONS = 100                     # number of rotations sampled in parameter graphs
