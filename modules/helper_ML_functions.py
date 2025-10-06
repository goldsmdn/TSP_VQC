#Classical ML functions

import torch.nn as nn
import torch
#from classes.MyDataLogger import MySubDataLogger

def find_device() -> torch.device:
    """find out if we are using a GPU or CPU"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return(device)

def evaluate_model(model:nn.Module, shots:int)-> dict:
    """store the bits strings from the model in a dictionary"""
    bits = model.find_bits()
    device = find_device()
    input = torch.zeros(shots, bits).to(device)
    binary_list = model(input).cpu()
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
                       **kwargs,
                       )-> tuple:
    """Prepare for training by setting up the target, criterion, and optimizer"""
    target = torch.tensor(0.0, requires_grad=True)
    criterion = nn.L1Loss()
    if optimizer in ['Adam', 'Adam+X',]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(kwargs['momentum'], 0.999))
    elif optimizer in ['SGD', 'SGD+X',]:
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
                print_results:bool=False,
                print_frequency:int=10) -> tuple:
    """Train the model for a number of epochs"""
    model_output = model(my_input)
    lowest_cost = float(model_output.min())
    index_list, loss_history, lowest_history = [], [], []
    epoch_lowest_cost_found = 0
    for epoch in range(num_epochs):
        index_list.append(epoch)
        model_output = model(my_input)
        loss = criterion(model_output.mean(), target)
        loss_history.append(float(loss))
        loss.backward()
        optimizer.step()
        epoch_min = float(model_output.min())
        if epoch_min < lowest_cost:
            lowest_cost = epoch_min
            epoch_lowest_cost_found = epoch
        lowest_history.append(lowest_cost)
        if print_results:
            if epoch % print_frequency == 0:
                print(
                    f'Epoch {epoch}, Average cost: {loss:.3f}', 
                    f'Epoch min cost:{epoch_min:.3f}, Lowest Cost to date: {lowest_cost:.3f}'
                    )
                # Check gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f'Epoch {epoch}, {name} grad: {param.grad.norm():.2f}')
                    else:
                        print(f'Epoch {epoch}, {name} grad is None')
        optimizer.zero_grad()
    
    return lowest_cost, epoch_lowest_cost_found, index_list, loss_history, lowest_history

def set_up_input_no_hot_start(sdl,
                              device: torch.device,
                              )-> torch.Tensor:    
    """if ML and no Hot Start set the initial input to zero OR 0.5, depending on the mode"""
    if sdl.mode in [8, 18]:
        #input is all zeros
        unrepeated_input = torch.full((1,sdl.qubits), 0).float().to(device)
    elif sdl.mode in [9, 19]:
        #input is all 0.5
        unrepeated_input = torch.full((1,sdl.qubits), 0.5).float().to(device)
    my_input = unrepeated_input.repeat(sdl.shots, 1).requires_grad_(True)
    return(unrepeated_input, my_input)

