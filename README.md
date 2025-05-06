# TSP_VQC
This note book provides the code to solve the Travelling Salesman problem (TSP) with a Variational Quantum Circuit, and a quantum inspired classical machine learning model.  The work is described in more detail in an [article](h.ttps://www.overleaf.com/project/67fd29849d97c8adf9efcd7b)  

## Process overview

An overview of the process is shown below.  In summary:
- TSP networks are stored in the `networks` folder, either loaded from external sources, or created automatically by [`make_data.ipynb`](make_data.ipynb).  
- runs can be executed manually by [`manual_runs_ML.ipynb`](manual_runs_ML.ipynb) for classical ML and [`manual_runs_VQC.ipynb`](manual_runs_VQC.ipynb) for quantum.  These allow an interactive environment for simple experiments.  In manual executions the control parameters are read from the a configuration data in [`modules/config.py`](/modules/config.py)
- most runs are executed automatically by [`auto_runs.ipynb`](auto_runs.ipynb)
- in any cases results data is updated to the ['results.csv'](/results/results.csv) file, and to sub-run specific results files and graphs
- each execution of data causes a `run-id` to be created, and each different set of configuration data causes a `sub-id` to be created.  
- data is analysed by [`show_results.ipynb`](show_results.ipynb)

![Image overview](/images/flowchart.png)

## Notebooks provided

### Data execution
The following Jupyter notebooks are provided for data execution:
 - [`auto_runs.ipynb`](auto_runs.ipynb): responsible for executing automatic runs, reading configuration data from [`control_parameters.csv`](/control/control_parameters.csv)
 - [`manual_runs_ML.ipynb`](manual_runs_ML.ipynb): responsible for executing manual runs of the classical ML model, reading configuration data from [`modules/config.py`](/modules/config.py)
 - [`manual_runs_VQC.ipynb`](manual_runs_VQC.ipynb): responsible for executing manual runs of the quantum machine learning model, reading configuration data from [`modules/config.py`](/modules/config.py)

### Network creation
The following Jupyter notebooks are provided for create networks for testing.  The networks are stored in the `networks` folder.
- [`make_data.ipynb`](make_data.ipynb): responsible for setting up new networks

### Data analysis
The following Jupyter notebooks are provided for data analysis:
- [`show_results.ipynb`](show_results.ipynb): responsible for analysing the results stored in the [`result/results.csv`](result/results.csv) file
- [`resource_requirements.ipynb`](resource_requirements.ipynb): calculates the number of qubits needed for each formulation
- [`hot_start_analysis.ipynb`](hot_start_analysis.ipynb): compares the Hamming distance of the hot start binary string to the binary string of the optimum solution

## Python modules
The following modules are provided in the modules folder:

### Helper functions
- [`helper_functions_tsp.py`](/modules/helper_functions_tsp.py): general helper functions
- [`graph_functions.py`](/modules/graph_functions.py): plots graphs
- [`helper_ML_functions.py`](/modules/helper_ML_functions.py): specific to classical machine learning model

### Test functions
A full suite of over 70 test Unit Test cases is provided and executed automatically using PyTest on each push to the repository
- [`test_ML_functions.py`](/modules/test_ML_functions.py): unit test cases for classical machine learning
- [`test_quantum_functions.py`](/modules/test_quantum_functions.py): unit test cases for quantum machine learning
- [`test_tsp_helper.py`](/modules/test_quantum_functions.py): general unit test cases

## Pythcn classes
The following object orientated code is provided:
- [`LRUCacheUnhashalble.py`](/classes/LRUCacheUnhashalble.py): handles caches of bit string evaluations
- [`MyDataLogger.py`](/classes/MyDataLogger.py): handles logging of data results including updating `results.txt`, and sub-run specific data summaries and graphs.  This module is object orientated, with objects for a parent `run-id` and child `sub-id`.
- [`MyModel.py`](classes/MyModel.py): responsbile for classical machine learing PyTorch modules

## Key coding points

### Optimisers
The optimiser is chosen setting the constant `GRADIENT_TYPE`.  For quantum two optimisers have been coded by hand:
 - `parameter_shift` which uses the fact that qubit rotations are trigonometric functions to find an analytical expression for the gradient.  Please see [Pennylane documentation](https://pennylane.ai/qml/glossary/parameter_shift) for a full description of parameter shift.
 - `SPSA` is an algorithm of optimisation invented by James C. Spall specially useful for noisy cost functions and the ones which the exact gradient is not available. Please see a [blog](https://www.geeksforgeeks.org/spsa-simultaneous-perturbation-stochastic-approximation-algorithm-using-python/) for a description of SPSA code that was modified.