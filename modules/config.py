import numpy as np
CONTROL_FILE = 'control/control_parameters.csv'
ENCODING = 'utf-8-sig'              # Encoding of csv file
GRAPH_DIRECTORY = 'graphs/'         # location of graph directory
LOCATIONS = 5                       # number of locations to be visited
ROTATIONS = 100                     # number of rotations sampled in parameter graphs
SHOTS = 1024                        # shots used for each call of the quantum circuit

MODE = 2                            # MODE = 1 - rxgate, rygate, cnot gates
                                    # MODE = 2 - rxgate, XX gates -can be used with Hot Start

ITERATIONS = 50                     # updates, or iterations
PRINT_FREQUENCY = 10                # how often results are printed out
GRAY = True                         # Use Gray codes
HOT_START = True                    # Make a hot start
VERBOSE = False                     # controls how much is printed
GRADIENT_TYPE = 'SPSA'              # controls the optimiser used
                                    # 'parameter_shift' - default
                                    # 'SPSA' is a stochastic gradient descent

S = 0.5                             # parameter for parameter shift.  Default is 0.5
ETA = 0.02                          # eta - learning rate for parameter shift
ALPHA = 0.602                       # constant that controls the learning rate for SPSA decays
GAMMA = 0.101                       # constant that determines how quickly the SPSA perturbation decays
C = np.pi/10                        # initial CK for SPSA
BIG_A = 50                          # A for SPSA

CACHE_MAX_SIZE = 500_000             #maximum size of the cache.

DATA_SOURCES = {  4 : {'file' : 'data/four_d.txt', 'best' : 21},
                  5 : {'file' : 'data/five_d.txt', 'best' : 19},
                 11 : {'file' : 'data/dg11_d.txt', 'best' : 253},
                 15 : {'file' : 'data/p01_d.txt',  'best' : 291},
                 17 : {'file' : 'data/gr17_d.txt', 'best' : 2085},
                 26 : {'file' : 'data/fri26_d.txt', 'best' : 699},
                 42 : {'file' : 'data/dantzig42_d.txt', 'best' : 33_523},
                }

CHANGE_EACH_PARAMETER = True        # Iterate through each parameter in the circuit
PLOT_PARAMETER_EVALUATION = True    # Plot the evaluation of each parameter

SLICES = [1, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.05]                      
                                    # Slices to use when calculating the gradient
                                    #[1, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.05] 
                                    # For example, 0.2 means that the best 20% 
                                    # of distances found is included in the average.
                                 
DECODING_FORMULATION = 'new'        # 'original' or 'new' - new is forumlation from paper
