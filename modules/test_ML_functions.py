import torch
from pathlib import Path
import numpy as np

from classes.MyDataLogger import MyDataLogger, MySubDataLogger

from classes.MyModel import estimate_cost_fn_gradient
from modules.helper_ML_functions import find_device
from modules.helper_functions_tsp import (load_dict_from_json, 
                                          read_file_name,
                                          validate_distance_array,
                                          cost_fn_fact, 
                                          cost_fn_tensor,
                                          )

from modules.config import NETWORK_DIR, DATA_SOURCES

def test_estimate_gradient():
    """checks gradient estimation against pre-worked example"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 5
    sdl.formulation = 'original'
    sdl.gray = False
    device = find_device()
    my_input = torch.tensor([[1., 0., 0., 1., 0.]]).float().to(device)
    sources_filename = Path(NETWORK_DIR).joinpath(DATA_SOURCES)
    data_source_dict = load_dict_from_json(sources_filename)
    filename = read_file_name(str(sdl.locations), data_source_dict)
    filepath = Path(NETWORK_DIR).joinpath(filename)
    distance_array = np.genfromtxt(filepath)
    validate_distance_array(distance_array, sdl.locations)
    cost_fn = cost_fn_fact(sdl, distance_array,)
    output = cost_fn_tensor(my_input, cost_fn).to(device)
    actual_result = estimate_cost_fn_gradient(my_input, output, cost_fn).float().to(device)
    expected_result = torch.tensor([[ -8.0, 6.0,  6.0,  -6.0,  0.0]]).float().to(device)
    assert torch.allclose(actual_result, expected_result, atol=1e-4)

def test_estimate_gradient_2():
    """checks gradient estimation against pre-worked example for 2*5 input"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 5
    sdl.formulation = 'original'
    sdl.gray = False
    device = find_device()
    my_input = torch.tensor([[1., 0., 0., 1., 0.], 
                          [1., 0., 0., 1., 0.]]).float().to(device)
    sources_filename = Path(NETWORK_DIR).joinpath(DATA_SOURCES)
    data_source_dict = load_dict_from_json(sources_filename)
    filename = read_file_name(str(sdl.locations), data_source_dict)
    filepath = Path(NETWORK_DIR).joinpath(filename)
    distance_array = np.genfromtxt(filepath)
    validate_distance_array(distance_array, sdl.locations)
    cost_fn = cost_fn_fact(sdl, distance_array) 
    output = cost_fn_tensor(my_input, cost_fn).to(device)
    actual_result = estimate_cost_fn_gradient(my_input, output, cost_fn).float().to(device)
    expected_result = torch.tensor([[ -8.0, 6.0,  6.0,  -6.0,  0.0], 
                                    [ -8.0, 6.0,  6.0,  -6.0,  0.0]]).float().to(device)
    assert torch.allclose(actual_result, expected_result, atol=1e-4)

def test_estimate_gradient_3():
    """checks gradient estimation against a second pre-worked example for 2*5 input"""
    datalogger = MyDataLogger()
    sdl = MySubDataLogger(runid = datalogger.runid)
    sdl.locations = 5
    sdl.formulation = 'original'
    sdl.gray = False
    device = find_device()
    my_input = torch.tensor([[1., 0., 0., 1., 0.], 
                             [0., 0., 0., 1., 0.]]).float().to(device)
    sources_filename = Path(NETWORK_DIR).joinpath(DATA_SOURCES)
    data_source_dict = load_dict_from_json(sources_filename)
    filename = read_file_name(str(sdl.locations), data_source_dict)
    filepath = Path(NETWORK_DIR).joinpath(filename)
    distance_array = np.genfromtxt(filepath)
    validate_distance_array(distance_array, sdl.locations)
    cost_fn = cost_fn_fact(sdl, distance_array) 
    output = cost_fn_tensor(my_input, cost_fn).to(device)
    actual_result = estimate_cost_fn_gradient(my_input, output, cost_fn).float().to(device)
    expected_result = torch.tensor([[ -8.0,  6.0,  6.0, -6.0,  0.0], 
                                    [ -8.0, -4.0, -4.0,  4.0, -2.0]]).float().to(device)
    assert torch.allclose(actual_result, expected_result, atol=1e-4)