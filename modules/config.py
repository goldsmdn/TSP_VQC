import numpy as np
CONTROL_FILE = 'control/control_parameters.csv'
ENCODING = 'utf-8-sig'              # Encoding of csv file
GRAPH_DIRECTORY = 'graphs/'         # location of graph directory
LOCATIONS = 5                       # number of locations to be visited
ROTATIONS = 100                     # number of rotations sampled in parameter graphs
SHOTS = 1024                        # shots used for each call of the quantum circuit
AVERAGE_SLICE = 1                   # controls the amount of data to be included in the average.  
                                    # Default  = 1 - all data
                                    # For example, 0.2 means that the lowest 20% 
                                    # of distances found is included in the average.

MODE = 2                            # MODE = 1 - rxgate, rygate, cnot gates
                                    # MODE = 2 - rxgate, XX gates -can be used with Hot Start

ITERATIONS = 50                     # updates, or iterations
PRINT_FREQUENCY = 5                 # how often results are printed out
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