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
            sign = 2 * (old_bit - 0.5)
            my_input_clone[i,j] = 1 - old_bit
            new_output = cost_fn_tensor(my_input_clone[[i]], cost_fn).to(device)
            gradient_est[i, j] = (output[i] - new_output) * sign
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
        #mean = torch.mean(output).to(device).requires_grad_(True)
        #return mean
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Read the gradient from the forward pass
        grad_cost_fn = None
        return ctx.grad, grad_cost_fn
    
class MySine(nn.Module):
    """return a sine function symmetric about 0.5"""
    def __init__(self):
        super(MySine, self).__init__()  # Initialize parent class

    def forward(self, x):
        rads = ((x - 0.5) * torch.tensor(math.pi))
        return 0.5 * (1 + torch.sin(rads)) 
    
class Sample_Binary(nn.Module):
    """return probability in forward, linear backwards"""
    def __init__(self):
        super(Sample_Binary, self).__init__()  # Initialize parent class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sampled = (torch.rand_like(x) < x).float()  # 0/1 values as float
        output = sampled.to(torch.int)
        return x + (output - x).detach()

class BinaryToCost(nn.Module):
    """convert a bit string to a cost in forwards, estimate gradient backwards"""
    def __init__(self, cost_fn:Callable[[list], int]):
        super(BinaryToCost, self).__init__() # Intialize parent class
        self.cost_fn = cost_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """calculate cost forwards"""
        x = CostFunction.apply(x, self.cost_fn)
        return x

class MyModel(nn.Module):
    def __init__(self, bits:int, layers:int, std_dev:float, cost_fn:Callable[[list], int]):
        """initialize the model"""
        super(MyModel, self).__init__()
        self.bits = bits
        self.layers = layers
        self.std_dev = std_dev
        self.cost_fn = cost_fn
        self.fc1 = nn.Linear(in_features=bits, out_features=bits)
        self.act1 = MySine()
        if self.layers == 2:
            self.fc2 = nn.Linear(in_features=bits, out_features=bits)
            self.act2 = MySine()
        elif self.layers > 2:
            raise Exception(f'Only 2 layers are coded for. {self.layers} is to many')
        self.sample = Sample_Binary()
        self.cost = BinaryToCost(self.cost_fn)
        self.generate_weights_and_biases()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.layers == 2:
            x = self.fc2(x)
            x = self.act2(x)
        elif self.layers > 2:
            raise Exception(f'Only 2 layers are coded for.  {self.layers} is too many')
        x = self.sample(x)
        x = self.cost(x)
        return(x)
    
    def generate_weights_and_biases(self):
        """generate random weights and biases"""
        weights_zeros = torch.zeros(self.bits, self.bits)
        new_weights = torch.eye(self.bits) + torch.normal(mean=weights_zeros, 
                                           std=self.std_dev)
        bias_zeros = torch.zeros(self.bits)
        new_bias = torch.normal(mean=bias_zeros, std=self.std_dev)

        self.fc1.weight = torch.nn.Parameter(new_weights)
        self.fc1.bias = torch.nn.Parameter(new_bias)
        if self.layers == 2:
            self.fc2.weight = torch.nn.Parameter(new_weights)
            self.fc2.bias = torch.nn.Parameter(new_bias)
        elif self.layers > 2:
            raise Exception(f'Only 2 layers are coded for.  {self.layers} is to many')