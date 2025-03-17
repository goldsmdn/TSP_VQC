#Classical ML functions

import torch.nn as nn
import torch

def find_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return(device)

def evaluate_model(model, shots):
    bits = model.find_bits()
    device = find_device()
    input = torch.zeros(shots, bits).to(device)
    binary_list = model(input).cpu()

    #print(binary_list)
    counts = {}
    for bit_list in binary_list:
        string = ''
        for item in bit_list:
            string += str(item)
        if string in counts:
            counts[string] += 1
        else:
            counts[string] = 1
    return(counts)

def get_ready_to_train(model:nn.Module,
                       optimizer:str, 
                       lr:float, 
                       weight_decay:float,
                         **kwargs)-> tuple:
    """Prepare for training by setting up the target, criterion, and optimizer"""
    target = torch.tensor(0.0, requires_grad=True)
    criterion = nn.L1Loss()
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), momentum=kwargs['momentum'], lr=lr, weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {optimizer} not recognized')
    return(target, criterion, optimizer)

def train_model(num_epochs: int,
                model:nn.Module, 
                my_input:torch.Tensor, 
                target:torch.Tensor, 
                criterion:nn.Module,
                optimizer:torch.optim.Optimizer, 
                print_results:bool=False) -> tuple:
    """Train the model for a number of epochs"""
    output = model(my_input)
    lowest_cost = float(output)
    epoch_history = []
    loss_history = []
    epoch_lowest_cost_found = 0

    for epoch in range(num_epochs):
        epoch_history.append(epoch)
        model_output = model(my_input)
        loss = criterion(model_output, target)
        loss_history.append(float(loss))
        loss.backward()
        optimizer.step()
        if float(model_output ) < lowest_cost:
            lowest_cost = float(loss)
            epoch_lowest_cost_found = epoch
        if print_results:
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Cost: {loss:.3f}, Lowest Cost to date =  {lowest_cost:.3f}")
                # Check gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f'Epoch {epoch}, {name} grad: {param.grad.norm():.2f}')
                    else:
                        print(f'Epoch {epoch}, {name} grad is None')
        optimizer.zero_grad()
    
    return lowest_cost, epoch_lowest_cost_found, epoch_history, loss_history

