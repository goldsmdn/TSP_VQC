# TSP_VQC
Travelling Salesman problem with a Variational Quantum Circuit.

## Process overview

An overview of the process is shown below:

![Image overview](/images/flowchart.png)

## Notebooks provided

The following Jupyter notebooks are provided for data execution:
 - [`auto_runs.ipynb`](auto_runs.ipynb): responsible for executing automatic runs, reading configuration data from [`control/control_parameters.csv`](/control/control_parameters.csv)
 - [`manual_runs_ML.ipynb`](manual_runs_ML.ipynb): responsible for executing manual runs of the classical ML model, reading configuration data from [`modules/config.py`](/modules/config.py)
 - [`manual_runs_VQC.ipynb`](manual_runs_VQC.ipynb): responsible for executing manual runs of the quantum machine learning model, reading configuration data from [`modules/config.py`](/modules/config.py)

The following Jupyter notebooks are provided for data set up:
- [`make_data.ipynb`](make_data.ipynb): responsible for setting up new networks

The following Jupyter notebooks are provided for data analysis:
- [`show_results.ipynb`](show_results.ipynb): responsible for analysing the results stored in the [`result/results.csv`](result/results.csv) file
- [`resource_requirements.ipynb`](resource_requirements.ipynb): calculates the number of qubits needed for each formulation
- [`hot_start_analysis.ipynb`](hot_start_analysis.ipynb): compares the Hamming distance of the hot start binary string to the binary string of the optimum solution

## Gradient calculation
Two methods are available and can be chosen by setting the constant `GRADIENT_TYPE` to
 - `parameter_shift` or
 - `SPSA`

### Parameter shift 
Parameter shift uses the fact that qubit rotations are trigonometric functions to find an analytical expression for the gradient

Please see [Pennylane documentation](https://pennylane.ai/qml/glossary/parameter_shift) for a full description of parameter shift.

### SPSA
SPSA is an algorithm of optimisation invented by James C. Spall specially useful for noisy cost functions and the ones which the exact gradient is not available.

Please see a [blog](https://www.geeksforgeeks.org/spsa-simultaneous-perturbation-stochastic-approximation-algorithm-using-python/) for a description of SPSA code that was modified.