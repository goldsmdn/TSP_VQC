#General helper functions that could be used for other projects, not just Travelling Salesman Problem (TSP)

from modules.config import VALID_QUBIT_LOOPS

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

def find_valid_device_loop(qubits:int, target ) -> list:
    """read the valid qubit loops as a list from the configuration file"""
    if target == 'local':
    # don't need a bespoke qubit list
        output_list = [i for i in range(qubits)]
    else:
        output_list = VALID_QUBIT_LOOPS[target][qubits]
    return output_list

def find_logical_to_physical_dictionary(qubits:int, target) -> dict:
    """return a dictionary showing the looking up from logical to physical qubit"""
    my_list = find_valid_device_loop(qubits, target)
    output_dict = convert_list_to_dictionary(my_list)
    return output_dict

def convert_physical_to_logical_bit_string(input_bitstring:list, qubits:int, target) -> list:
    """finds the permutation to be applied to the output bit string"""
    #print(f'{input_bitstring=}, {qubits=}')
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
    #output_list = output_list[:qubits] #only send the items for logical qubits. 
    return output_list

def find_qubits_measured(qubits:int, target) -> int:
    return len(find_valid_device_loop(qubits, target))
    
    