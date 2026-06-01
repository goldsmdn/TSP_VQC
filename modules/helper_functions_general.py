#General helper functions that could be used for other projects, not just Travelling Salesman Problem (TSP)

import json
import csv

from itertools import count

from modules.config import VALID_QUBIT_LOOPS

def format_boolean(string_input: str)->bool:
    """Convert a string to a boolean value"""
    if string_input == 'TRUE':
        output = True
    elif string_input == 'FALSE':
        output = False
    else:
        raise Exception(f'Unexpected boolean value {string_input}')
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

    return(formatted_string)    

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
    with open( filename, 'r', encoding=encoding) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dict[next(index)] = row
    return(dict)

def validate_list_for_duplicates(input_list:list) -> bool:
    """Validate that a list does not contain duplicates"""
    if len(input_list) != len(set(input_list)):
        return False
    else:
        return True

def convert_list_to_dictionary(input_list:list) -> dict:
    """Convert a list to a dictionary with the list elements as values and the keys as the index of the element in the list"""
    duplicates = validate_list_for_duplicates(input_list)
    if duplicates is False:
        raise Exception(f'Qubit list {input_list} contains duplicates, not a valid input')
    output_dict = {}
    for key, item in enumerate(input_list):
        output_dict[key] = item
    return output_dict

def find_valid_device_loop(qubits:int, target:str ) -> list:
    """read the valid qubit loops as a list from the configuration file"""
    #print(f'Finding valid device loop for {qubits} qubits and target {target}')
    if target in ['local_aws', 'local_qiskit', 'ml']:
    # don't need a bespoke qubit list
        output_list = [i for i in range(qubits)]
    else:
        output_list = VALID_QUBIT_LOOPS[target][qubits]
    return output_list

def find_logical_to_physical_dictionary(qubits:int, target:str) -> dict:
    """return a dictionary showing the looking up from logical to physical qubit"""
    my_list = find_valid_device_loop(qubits, target)
    output_dict = convert_list_to_dictionary(my_list)
    return output_dict

def convert_physical_to_logical_bit_string(
        input_bitstring:list, 
        qubits:int, 
        target:str) -> list:
    """finds the permutation to be applied to the output bit string"""
    output_list = []
    qubit_list = find_valid_device_loop(qubits, target)
    # remove last qubit - highest physical qubit
    qubit_list = qubit_list[:qubits]
    sorted_list = sorted(qubit_list) #only process the qubits mapped to logical qubits. 
    logical_to_physical_dict = find_logical_to_physical_dictionary(qubits, target)
    physical_to_logical_dict = {value: key for key, value in logical_to_physical_dict.items()}   
    for i, item in enumerate(sorted_list):
            physical_qubit = item
            logical_qubit = physical_to_logical_dict[physical_qubit]
            output_list.append(input_bitstring[logical_qubit])
    return output_list

def find_qubits_measured(qubits:int, target:str) -> int:
    return len(find_valid_device_loop(qubits, target))