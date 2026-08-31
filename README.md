# TSP_VQC
This note book provides the code to solve the Travelling Salesman problem (TSP) with a Variational Quantum Circuit, and a quantum inspired classical machine learning model.  Full documentation can be found in the [Read_the_Docs](https://goldsmdn.github.io/TSP_VQC/)

The work is described in more detail in an [article](https://arxiv.org/abs/2512.06523) 

Note:  Circuit 6 in this repo is re-numbered as Circuit 5 in the article.  Circuit 2 is renumbered as 2a, and circuit 22 and 2b.

## Getting started 

### Install the repository locally
Clone the repository to a suitable location on your computer using the following command:
```
git clone https://github.com/goldsmdn/TSP_VQC

``` 
### Install uv

Please see the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)

### Set up dependencies using uv
Use the following command run from the TSP_VQC directory
```
uv venv
source .venv/Scripts/activate
uv pip install numpy pytest graycode qiskit qiskit_aer torch
uv pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints matplotlib
uv pip install qiskit_ibm_runtime pandas seaborn amazon-braket-sdk>=1.50.0 ipython nevergrad \
          numba>=0.57.0 llvmlite>=0.40.0
uv pip install ipykernel notebook pylatexenc
uv pip install torchviz graphviz
uv pip install ortools
```

## Running the notebooks
To run one of the notebooks, for example `manual_runs_ML.ipynb` enter

```
jupyter notebook manual_runs_ML.ipynb

```
Alternatively you can run in the VS code development environment, setting the Python interpreter to Base.


## Process overview

An overview of the process is shown below.  In summary:
- TSP networks are stored in the [`networks`](/networks) folder, either loaded from external sources, or created automatically by [`make_data.ipynb`](make_data.ipynb).  
- runs can be executed manually by 
    - [`manual_runs_ML.ipynb`](manual_runs_ML.ipynb) for classical ML 
    - [`manual_runs_VQC.ipynb`](manual_runs_VQC.ipynb) for quantum.  These allow an interactive environment for simple experiments.  In manual executions the control parameters are read from the a configuration data in [`modules/config.py`](/modules/config.py)
    - [`nevergrad_comparison.ipynb`](nevergrad_comparison.ipynb) for Nevergrad experiments
    - [`ORtools.ipynb`](ORtools.ipynb) for Google OR Tools optimiser
- most runs are executed automatically by 
    - [`auto_runs.ipynb`](auto_runs.ipynb) default option
    - [`auto_runs_AWS.ipynb`](auto_runs_AWS.ipynb) 
    - [`auto_runs_AWS_ng.ipynb`](auto_runs_AWS_ng.ipynb) AWS runs with Nevergrad optimisers like CMA-ES
- in any cases results data is updated to the [`results.csv`](/results/results.csv) file, and to sub-run specific results files and graphs
- each execution of data causes a `run-id` to be created, and each different set of configuration data causes a `sub-id` to be created.  
- data is analysed and graphs plotted by [`show_results.ipynb`](show_results.ipynb)

![Image overview](/images/flowchart.png)

## Notebooks provided

### Data execution
The following Jupyter notebooks are provided for data execution:
 - [`auto_runs.ipynb`](auto_runs.ipynb): responsible for executing automatic runs, reading configuration data from [`control_parameters.csv`](/control/control_parameters.csv)
 - [`auto_runs_AWS.ipynb`](auto_runs_AWS.ipynb) AWS runs with Q-SPSA or SPSA
 - [`auto_runs_AWS_ng.ipynb`](auto_runs_AWS_ng.ipynb) AWS runs with Nevergrad optimisers like CMA-ES
 - [`manual_runs_ML.ipynb`](manual_runs_ML.ipynb): responsible for executing manual runs of the classical ML model, reading configuration data from [`modules/config.py`](/modules/config.py)
 - [`manual_runs_VQC.ipynb`](manual_runs_VQC.ipynb): responsible for executing manual runs of the quantum machine learning model, reading configuration data from [`modules/config.py`](/modules/config.py)
 - [`ORtools.ipynb`](ORtools.ipynb) for Google OR Tools optimiser
 - [`nevergrad_comparison.ipynb`](nevergrad_comparison.ipynb) for Nevergrad experiments
 - [`monte_carlo.ipynb`](monte_carlo.ipynb): carries out Monte Carlo simulations by finding the best distance over a range of bit strings

### Network creation
The following Jupyter notebooks are provided for create networks for testing.  The networks are stored in the [`networks`](/networks)  folder.
- [`make_data.ipynb`](make_data.ipynb): responsible for setting up new networks

### Data analysis
The following Jupyter notebooks are provided for data analysis:
- [`show_results.ipynb`](show_results.ipynb): responsible for analysing the results stored in the [`result/results.csv`](result/results.csv) file
- [`plot_data.ipynb`](plot_data.ipynb): resonsible for creating bespoke graphs of individual runs and plots anomolous network with 42 locations
- [`resource_requirements.ipynb`](resource_requirements.ipynb): calculates the number of qubits needed for each formulation
- [`hot_start_analysis.ipynb`](hot_start_analysis.ipynb): compares the Hamming distance of the hot start binary string to the binary string of the optimum solution
- [`bit_strings_ranked_by_distance.ipynb`](bit_strings_ranked_by_distance.ipynb): Plots a graph of the solution quality by ordered bit string

## Python modules

The following modules are provided in the modules folder:

### Configuation file
- [`config.py`](/modules/config.py) Primary configuration file
- [`config_graphs.py](/modules/config_graphs.py)

### Helper functions
- [`graph_functions.py`](/modules/graph_functions.py): plots graphs
- [`helper_functions_general.py`](/modules/helper_functions_general.py):  general helper functions
- [`helper_functions_nevergrad`](/modules/helper_functions_nevergrad.py): functions to support Nevergrad experiments
- [`helper_functions_tsp.py`](/modules/helper_functions_tsp.py): helper functions particularly written for the Travelling Salesman problem
- [`helper_results.py`](/modules/helper_results.py): functions to help plot results
- [`helper_ML_functions.py`](/modules/helper_ML_functions.py): specific to classical machine learning model
- [`quantum_circuits.py`](/modules/quantum_circuits.py): functions to build and print out quantum circuits

### Test functions
A full suite of over 100 test Unit Test cases is provided and executed automatically using PyTest on each push to the repository
- [`test_ML_functions.py`](/modules/test_ML_functions.py): unit test cases for classical machine learning
- [`test_quantum_functions.py`](/modules/test_quantum_functions.py): unit test cases for quantum machine learning
- [`test_tsp_helper.py`](/modules/test_quantum_functions.py): general unit test cases

## Python classes
The following object orientated code is provided:
- [`LRUCacheUnhashable.py`](/classes/LRUCacheUnhashable.py): handles caches of bit string evaluations
- [`MyDataLogger.py`](/classes/MyDataLogger.py): handles logging of data results including updating `results.txt`, and sub-run specific data summaries and graphs.  This module is object orientated, with objects for a parent `run-id` and child `sub-id`.
- [`MyModel.py`](classes/MyModel.py): responsbile for classical machine learing PyTorch modules

## Contributing
Contributions to the repository are very welcome.  Please raise an issue if you have any problems, and feel free to contact me.

## Key coding points

### Optimisers
The optimiser is chosen setting the constant `GRADIENT_TYPE`.  For quantum three optimisers bespoke coding is provided:
 - `parameter_shift` which uses the fact that qubit rotations are trigonometric functions to find an analytical expression for the gradient.  Please see [Pennylane documentation](https://pennylane.ai/qml/glossary/parameter_shift) for a full description of parameter shift.
 - `SPSA` is an algorithm of optimisation invented by James C. Spall specially useful for noisy cost functions and the ones which the exact gradient is not available. Please see a [blog](https://www.geeksforgeeks.org/spsa-simultaneous-perturbation-stochastic-approximation-algorithm-using-python/) for a description of SPSA code that was modified.
 - `Q-SPSA` - a variant of SPSA with few cost evaluations
 Additional optimisers can be tested with Nevergrad