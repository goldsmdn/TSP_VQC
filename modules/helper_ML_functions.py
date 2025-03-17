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



