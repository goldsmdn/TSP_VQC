from pathlib import Path

import numpy as np
import torch

from classes.MyModel import estimate_cost_fn_gradient
from modules.config import DATA_SOURCES, NETWORK_DIR
from modules.helper_functions_tsp import (
    cost_fn_fact,
    cost_fn_tensor,
    load_dict_from_json,
    read_file_name,
    validate_distance_array,
)
from modules.helper_ML_functions import find_device


def test_estimate_gradient():
    """Checks gradient estimation against pre-worked example"""
    locations = 5
    formulation = 'original'
    gray = False
    device = find_device()
    my_input = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]]).float().to(device)
    sources_filename = Path(NETWORK_DIR).joinpath(DATA_SOURCES)
    data_source_dict = load_dict_from_json(sources_filename)
    filename = read_file_name(str(locations), data_source_dict)
    filepath = Path(NETWORK_DIR).joinpath(filename)
    distance_array = np.genfromtxt(filepath)
    validate_distance_array(distance_array, locations)
    cost_fn = cost_fn_fact(
        locations=locations,
        gray=gray,
        formulation=formulation,
        distance_array=distance_array,
    )
    output = cost_fn_tensor(my_input, cost_fn).to(device)
    actual_result = (
        estimate_cost_fn_gradient(my_input, output, cost_fn).float().to(device)
    )
    expected_result = torch.tensor([[-8.0, 6.0, 6.0, -6.0, 0.0]]).float().to(device)
    assert torch.allclose(actual_result, expected_result, atol=1e-4)


def test_estimate_gradient_2():
    """Checks gradient estimation against pre-worked example for 2*5 input"""
    locations = 5
    formulation = 'original'
    gray = False
    device = find_device()
    my_input = (
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]])
        .float()
        .to(device)
    )
    sources_filename = Path(NETWORK_DIR).joinpath(DATA_SOURCES)
    data_source_dict = load_dict_from_json(sources_filename)
    filename = read_file_name(str(locations), data_source_dict)
    filepath = Path(NETWORK_DIR).joinpath(filename)
    distance_array = np.genfromtxt(filepath)
    validate_distance_array(distance_array, locations)
    cost_fn = cost_fn_fact(
        locations=locations,
        gray=gray,
        formulation=formulation,
        distance_array=distance_array,
    )
    output = cost_fn_tensor(my_input, cost_fn).to(device)
    actual_result = (
        estimate_cost_fn_gradient(my_input, output, cost_fn).float().to(device)
    )
    expected_result = (
        torch.tensor([[-8.0, 6.0, 6.0, -6.0, 0.0], [-8.0, 6.0, 6.0, -6.0, 0.0]])
        .float()
        .to(device)
    )
    assert torch.allclose(actual_result, expected_result, atol=1e-4)


def test_estimate_gradient_3():
    """Checks gradient estimation against a second pre-worked example for 2*5 input"""
    locations = 5
    formulation = 'original'
    gray = False

    device = find_device()
    my_input = (
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]])
        .float()
        .to(device)
    )
    sources_filename = Path(NETWORK_DIR).joinpath(DATA_SOURCES)
    data_source_dict = load_dict_from_json(sources_filename)
    filename = read_file_name(str(locations), data_source_dict)
    filepath = Path(NETWORK_DIR).joinpath(filename)
    distance_array = np.genfromtxt(filepath)
    validate_distance_array(distance_array, locations)
    cost_fn = cost_fn_fact(
        locations=locations,
        gray=gray,
        formulation=formulation,
        distance_array=distance_array,
    )
    output = cost_fn_tensor(my_input, cost_fn).to(device)
    actual_result = (
        estimate_cost_fn_gradient(my_input, output, cost_fn).float().to(device)
    )
    expected_result = (
        torch.tensor([[-8.0, 6.0, 6.0, -6.0, 0.0], [-8.0, -4.0, -4.0, 4.0, -2.0]])
        .float()
        .to(device)
    )
    assert torch.allclose(actual_result, expected_result, atol=1e-4)
