import torch.nn as nn
import torch
import math
from typing import Callable

from modules.helper_ML_functions import find_device
from modules.helper_functions_tsp import cost_fn_tensor

def estimate_cost_fn_gradient(my_input:torch.Tensor, 
                              output:torch.Tensor, 
                              cost_fn: Callable[[list], int]) -> torch.Tensor:
    """estimate the gradient of the cost function by 
    changing each bit in turn and calculating the difference in cost function.
    
    Parameters
    ----------
    my_input : torch.Tensor
        The input tensor
    output : torch.Tensor
        Contains a precalculated run of the cost function 
        for performance reasons
    cost_fn : Callable[[list], int]
        The cost function to be used

    Returns
    -------
    torch.Tensor
        The estimated gradient.
    """
    device = find_device()
    gradient_est = torch.zeros_like(my_input)  # Initialize with zeros for clarity

    dim0 = my_input.size(0)
    dim1 = my_input.size(1)

    my_input_clone = my_input.clone()  # Clone once, modify in-place
    for i in range(dim0): 
        for j in range(dim1):
            old_bit = my_input[i,j]
            sign = 2 * old_bit - 1  # Convert to -1 or 1
            my_input_clone[i,j] = 1 - old_bit
            new_output = cost_fn_tensor(my_input_clone[[i]], cost_fn).to(device)
            gradient_est[i, j] = (output[i] - new_output) / sign
            my_input_clone[i, j] = old_bit 
    return gradient_est

class CostFunction(torch.autograd.Function):
    """A custom autograd function to calculate the cost function and estimate the gradient"""
    @staticmethod
    def forward(ctx, input, cost_fn):
        # Save the gradient for the backward pass
        device = find_device()
        output = cost_fn_tensor(input, cost_fn).to(device)
        gradient_est = estimate_cost_fn_gradient(input, output, cost_fn)
        ctx.grad = gradient_est
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Read the gradient from the forward pass
        grad_cost_fn = None
        return ctx.grad, grad_cost_fn
    
class MySine(nn.Module):
    """Returns a sine function symmetric about 0.5"""
    def __init__(self):
        super(MySine, self).__init__()  # Initialize parent class
        self.register_buffer("pi", torch.tensor(math.pi))

    def forward(self, x):
        #rads = ((x - 0.5) * torch.tensor(math.pi))
        #return 0.5 * (1 + torch.sin(rads)) 
        return 0.5 * (1 + torch.sin((x - 0.5) * self.pi))
    
class Sample_Binary(nn.Module):
    """Return probability in forward, linear backwards"""
    def __init__(self):
        super(Sample_Binary, self).__init__()  # Initialize parent class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sampled = (torch.rand_like(x) < x).float()  # 0/1 values as float
        output = sampled.to(torch.int)
        return x + (output - x).detach()

class BinaryToCost(nn.Module):
    """Convert a bit string to a cost in forwards, estimate gradient backwards"""
    def __init__(self, cost_fn:Callable[[list], int]):
        super(BinaryToCost, self).__init__() # Intialize parent class
        self.cost_fn = cost_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """calculate cost forwards"""
        x = CostFunction.apply(x, self.cost_fn)
        return x

class MyModel(nn.Module):
    """A simple feedforward neural network model for TSP"""
    def __init__(self, sdl, cost_fn:Callable[[list], int]):
        """Initialize the model"""
        super(MyModel, self).__init__()
        self.bits = sdl.qubits
        self.layers = sdl.layers
        self.std_dev = sdl.std_dev
        self.cost_fn = cost_fn
        self.hot_start = sdl.hot_start
        self.mode = sdl.mode
        self.gradient_type = sdl.gradient_type
        if self.mode in [8, 9]:
            self.activation = MySine()
        elif self.mode in [18, 19]:
            self.activation = nn.Sigmoid()
        else:
            raise Exception(f'Mode {self.mode} is not supported')
        self._build_layers()

    def _init_weights(self, fc, first_layer:bool=False):
        """Helper: initialize weights depending on hot_start mode"""
        with torch.no_grad():
            if not self.hot_start:
                # Xavier initialization for sigmoid, SIREN for sine
                if self.gradient_type in ['SGD+X', 'Adam+X']:
                    if isinstance(self.activation, nn.Sigmoid):
                        gain = nn.init.calculate_gain("sigmoid")
                        nn.init.xavier_uniform_(fc.weight, gain=gain)
                        if fc.bias is not None:
                            nn.init.zeros_(fc.bias)
                    elif isinstance(self.activation, MySine):
                        # Sitzmann et al. 2020 (SIREN)
                        if first_layer:
                            # First layer: uniform(-1/num_inputs, 1/num_inputs)
                            fc.weight.uniform_(-1 / self.bits, 1 / self.bits)
                            if fc.bias is not None:
                                fc.bias.zero_()
                        else:
                            # Hidden layers: uniform(-sqrt(6 / num_inputs)/ω, sqrt(6 / num_inputs)/ω)
                            # Default ω₀ = 30 in the SIREN paper
                            w0 = 30.0
                            bound = (6 / self.bits) ** 0.5 / w0
                            fc.weight.uniform_(-bound, bound)
                            if fc.bias is not None:
                                fc.bias.uniform_(-bound, bound)
                    else:
                        raise Exception('Activation function {self.activation} not supported for SGD+X')

            else:
                fc.weight.copy_(torch.eye(self.bits))
                fc.weight.add_(torch.normal(mean=0.0, std=self.std_dev, size=fc.weight.shape))
                fc.bias.copy_(torch.normal(mean=0.0, std=self.std_dev, size=fc.bias.shape))

    def _build_layers(self):
        """Create layers fc1..fcN and act1..actN."""

        for i in range(1, self.layers + 1):
            fc = nn.Linear(in_features=self.bits, out_features=self.bits)
            self._init_weights(fc, first_layer=(i == 1))
            setattr(self, f"fc{i}", fc)
            setattr(self, f"act{i}", self.activation)
        self.sample = Sample_Binary()
        self.cost = BinaryToCost(self.cost_fn)

    def forward(self, x):
        """Define the forward pass"""
        for i in range(1, self.layers + 1):
            #iterate through the layers and create a forward pass
            fc = getattr(self, f'fc{i}')
            act = getattr(self, f'act{i}')
            x = fc(x)
            x = act(x)
        x = self.sample(x)
        x = self.cost(x)
        return(x)