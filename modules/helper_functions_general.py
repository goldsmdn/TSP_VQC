#General helper functions that could be used for other projects, not just Travelling Salesman Problem (TSP)

from modules.config import VA

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
        raise Exception(f'Qubit list contains duplicates, not a valid input')
    elif duplicates != True:
        raise Exception(f'Unexpected error validating list for duplicates, got {duplicates}')
    output_dict = {}
    for key, item in enumerate(input_list):
        output_dict[key] = item
    return output_dict

def process_output_bit_string(input_list, bit_string:str) -> str:
    """Process the output bit string from the quantum circuit to a list of integers"""
    output_list = []
    for char in bit_string:
        output_list.append(int(char))
    return output_list