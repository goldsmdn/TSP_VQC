# General helper functions that could be used for other projects, not just Travelling Salesman Problem (TSP)

import csv
import json
from itertools import count

from modules.config import VALID_QUBIT_LOOPS


def format_boolean(string_input: str) -> bool:
    """Convert a string to a boolean value"""
    if string_input == 'TRUE':
        output = True
    elif string_input == 'FALSE':
        output = False
    else:
        raise ValueError(f'Unexpected boolean value {string_input}')
    return output


def binary_string_format(binary_string: str, bin_len: str) -> str:
    """Format a binary string to remove the 0b prefix

    Parameters
    ----------
    binary_string : str
        A binary string
    bin_len : str
        Length of the binary string

    Returns
    -------
    formatted_string: str
        The binary string with the 0b prefix removed
    """
    formatted_string = binary_string[2:]
    formatted_string = formatted_string.zfill(bin_len)

    return formatted_string


def load_dict_from_json(filename: str) -> dict:
    """Loads a dictionary from a JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def read_index(filename: str, encoding: str) -> dict:
    """Reads CSV file and returns a dictionary

    Parameters
    ----------
    filename : str
        The filename of the CSV file.
    encoding : str
        The expected coding.  If this is missed
        get odd charactors at start of the file

    Returns
    -------
    dict : dict
        A dictionary with the contents on the CSV file
    """
    dict = {}
    index = count()
    with open(filename, 'r', encoding=encoding) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dict[next(index)] = row
    return dict


def validate_list_for_duplicates(input_list: list) -> bool:
    """Validate that a list does not contain duplicates"""
    if len(input_list) != len(set(input_list)):  # noqa: SIM103
        return False
    else:
        return True


def convert_list_to_dictionary(input_list: list) -> dict:
    """Convert a list to a dictionary with the list elements as values and the keys as the index of the element in the list"""
    duplicates = validate_list_for_duplicates(input_list)
    if duplicates is False:
        raise ValueError(
            f'Qubit list {input_list} contains duplicates, not a valid input'
        )
    output_dict = {}
    for key, item in enumerate(input_list):
        output_dict[key] = item
    return output_dict


def find_valid_device_loop(qubits: int, target: str) -> list:
    """read the valid qubit loops as a list from the configuration file"""
    if target in ['local_aws', 'local_qiskit', 'ml']:
        # don't need a bespoke qubit list
        output_list = [i for i in range(qubits)]
    else:
        output_list = VALID_QUBIT_LOOPS[target][qubits]
    return output_list


def find_logical_to_physical_dictionary(qubits: int, target: str) -> dict:
    """return a dictionary showing the look up from logical to physical qubit"""
    my_list = find_valid_device_loop(qubits, target)
    output_dict = convert_list_to_dictionary(my_list)
    return output_dict


def find_physical_to_logical_dictionary(qubits: int, target: str) -> dict:
    """return a dictionary showing the look up from physical to logical qubit"""
    output_dict = {}
    my_list = find_valid_device_loop(qubits, target)
    for i, item in enumerate(my_list):
        output_dict[item] = i
    return output_dict


def find_qubits_measured(qubits: int, target: str) -> int:
    return len(find_valid_device_loop(qubits, target))


def convert_binary_string_to_list(binary_string: str) -> list:
    """Convert a binary string to a list of integers"""
    return [int(bit) for bit in binary_string]


def convert_list_to_binary_string(input_list: list) -> str:
    """Convert a list of integers to a binary string"""
    return ''.join(str(bit) for bit in input_list)


def convert_physical_to_logical_bit_string(
    input_bitstring: list[int] | str, qubits: int, target: str
) -> list[int] | str:
    """converts from a physical bit string to a logical bit string, which may have one less bit"""
    # if the input is a string, convert it to a list
    if isinstance(input_bitstring, str):
        input_bitstring_list = convert_binary_string_to_list(input_bitstring)
    # if a list no change is need.
    elif isinstance(input_bitstring, list):
        input_bitstring_list = input_bitstring
    else:
        raise TypeError(f'incorrect type for {input_bitstring}')

    qubit_list = find_valid_device_loop(qubits, target)
    sorted_qubit_list = sorted(qubit_list)

    physical_to_logical_dict = find_physical_to_logical_dictionary(qubits, target)

    output_list = [0 for i in range(qubits)]
    for i in range(len(input_bitstring_list)):
        logical_qubit = physical_to_logical_dict[sorted_qubit_list[i]]
        if logical_qubit < qubits:
            # not all physical qubits are linked to a logical qubit
            output_list[logical_qubit] = input_bitstring_list[i]

    # if the input was a string, output a string
    if isinstance(input_bitstring, str):
        output = convert_list_to_binary_string(output_list)
    else:
        output = output_list

    return output
