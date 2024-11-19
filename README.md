# TSP_VQC
Travelling Salesman problem with a Variational Quantum Circuit

## Gradient calculation
Two methods are available and can be chosen by setting the constant `GRADIENT_TYPE` to
 - `'parameter_shift` or
 - `'SPSA`

### Parameter shift 
Parameter shift uses the fact that qubit rotations are trigonometric functions to find an analytical expression for the gradient

Please see [Pennylane documentation](https://pennylane.ai/qml/glossary/parameter_shift) for a full description of parameter shift.

### SPSA
SPSA is an algorithm of optimisation invented by James C. Spall specially useful for noisy cost functions and the ones which the exact gradient is not available.

Please see a [blog](https://www.geeksforgeeks.org/spsa-simultaneous-perturbation-stochastic-approximation-algorithm-using-python/) for a description of SPSA code that was modified.