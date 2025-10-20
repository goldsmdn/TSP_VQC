#Classical ML functions

import torch.nn as nn
import torch

def find_device() -> torch.device:
    """Find out if we are using a GPU or CPU"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return(device)

def evaluate_model(model:nn.Module, shots:int)-> dict:
    """Store the bits strings from the model in a dictionary"""
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

def get_ready_to_train(sdl,
                       model:nn.Module,
                       )-> tuple:   
    """Prepare for training by setting up the target, criterion, and optimizer"""
    target = torch.tensor(0.0, requires_grad=True)
    criterion = nn.L1Loss()
    if sdl.gradient_type in ['Adam', 'Adam+X',]:
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=sdl.lr,
                                     weight_decay=sdl.weight_decay, 
                                     betas=(sdl.momentum, 0.999)
                                     )
    elif sdl.gradient_type in ['SGD', 'SGD+X',]:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    momentum=sdl.momentum, 
                                    lr=sdl.lr, 
                                    weight_decay=sdl.weight_decay,
                                    )
    elif sdl.gradient_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                        lr=sdl.lr, 
                                        weight_decay=sdl.weight_decay,
                                        )
    else:
        raise ValueError(f'Optimizer {sdl.sdl.gradient_type} not recognized')
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
    """If ML and no Hot Start set the initial input to zero OR 0.5, depending on the mode"""
    if sdl.mode in [8, 18]:
        #input is all zeros
        unrepeated_input = torch.full((1,sdl.qubits), 0).float().to(device)
    elif sdl.mode in [9, 19]:
        #input is all 0.5
        unrepeated_input = torch.full((1,sdl.qubits), 0.5).float().to(device)
    my_input = unrepeated_input.repeat(sdl.shots, 1).requires_grad_(True)
    return(unrepeated_input, my_input)

def set_up_input_hot_start(sdl,
                           device: torch.device,
                           bin_hot_start_list:list,
                           print_results:bool=False,
                          )-> torch.Tensor:    
    """If ML and Hot Start set the initial input to the hot start data"""
    bin_hot_start_list_tensor = torch.tensor([bin_hot_start_list])
    unrepeated_input = bin_hot_start_list_tensor.float().to(device)
    my_input = unrepeated_input.repeat(sdl.shots, 1).requires_grad_(True)
    if print_results:
        print(f'bin_hot_start_list_tensor = {bin_hot_start_list_tensor}')
        print(f'The hot start distance is {sdl.hot_start_dist:.2f}, compared to a best distance of {sdl.best_dist:.2f}.')
    return(unrepeated_input, my_input)