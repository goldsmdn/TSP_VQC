#Class to handle data logging
from time import strftime
from pathlib import Path
import csv
from dataclasses import dataclass, asdict, field
from typing import Callable
from modules.helper_functions_general import find_qubits_measured

from modules.config import (RESULTS_DIR, 
                            RESULTS_FILE,
                            GRAPH_DIR,
                            CACHE_MAX_SIZE,
                            LOCATIONS, 
                            SHOTS,
                            MODE, 
                            ITERATIONS, 
                            GRAY, 
                            HOT_START,
                            GRADIENT_TYPE, 
                            S, 
                            ETA, 
                            ALPHA, 
                            GAMMA, 
                            C, 
                            BIG_A,    
                            DECODING_FORMULATION,
                            NUM_LAYERS, 
                            STD_DEV,
                            LR, 
                            MOMENTUM,
                            WEIGHT_DECAY,
                            SIMULATE_NOISE,
                            TARGET,
                            TARGETS,
                            MPS,
                            AWS,
                            )

from modules.graph_functions import cost_graph_multi
from modules.helper_functions_tsp import (
    find_sdk,
    find_sdk_from_dispatch_dir,
    find_params_per_qubit,
    find_multi_layers_allowed,
    validate_gradient_type,
    )        

from modules.helper_functions_general import (
    find_qubits_measured,
    format_boolean,
)

@dataclass
class MyDataLogger:
    """Parent - header information for a group of data runs"""
    runid: str = strftime('%Y%m%d-%H-%M-%S') # Define runid as a dataclass field
    graph_sub_path: Path = None
    results_sub_path: Path = None
    summary_results_filename: Path = None

    def __post_init__(self):
        """This method is called after __init__"""
        # Now we can call create_sub_graph_path and create_sub_results_path
        self.graph_sub_path = self.create_sub_graph_path()
        self.results_sub_path = self.create_sub_results_path()
        self.summary_results_filename = self.find_summary_results_filename()

    def create_sub_graph_path(self):
        """Create a folder for graphs"""
        graph_path = Path(GRAPH_DIR)
        graph_sub_path = Path.joinpath(graph_path, self.runid)
        graph_sub_path.mkdir(parents=True, exist_ok=True)
        return graph_sub_path
    
    def create_sub_results_path(self):
        """Create a folder for results"""
        results_path = Path(RESULTS_DIR)
        results_sub_path = Path.joinpath(results_path, self.runid)
        results_sub_path.mkdir(parents=True, exist_ok=True)
        return results_sub_path
    
    def find_summary_results_filename(self):
        """Create the filepath for the summary results""" 
        results_path = Path(RESULTS_DIR)
        summary_results_filename = Path.joinpath(results_path, RESULTS_FILE)
        return summary_results_filename
    
@dataclass
class MySubDataLogger(MyDataLogger):
    """Child details of each data run"""
    #file details
    subid:str = None
    detailed_results_filename: Path = None
    graph_filename: Path = None
    #general inputs
    quantum: bool = None  # Fixed: Use 'bool' instead of 'Bool'
    locations: int = None
    slice: float = 1.0
    shots: int = None 
    mode: str = None
    iterations: int = None
    gray: bool = None
    hot_start: bool = None 
    gradient_type: str = None 
    formulation: str = None
    #ml specific set up
    layers: int = None
    std_dev: float = None
    lr: float = None
    weight_decay: float = None 
    momentum: float = None
    #quantum specific input
    alpha: float = None #default= 0.602 
    big_a: float = None #default= 0.50 
    c: float = None     #default= 0.314
    eta: float = None   #default= 0.02 
    gamma: float = None #default= 0.5
    s: float = None 
    #calculated results
    qubits: int = None                 #number of qubits / binary variables needed
    elapsed: float = None 
    hot_start_dist: float = None
    best_dist_found: float = None
    best_dist: float = None
    iteration_found: int = None
    #Cache statistics
    cache_max_size:int = None
    cache_items:int = None
    cache_hits:int = None 
    cache_misses:int = None
    #detailed_results
    index_list: list = field(default_factory=list)  # Use default_factory for mutable defaults
    average_list: list = field(default_factory=list)
    lowest_list: list = field(default_factory=list)
    sliced_list: list = field(default_factory=list)
    #results for graphing
    average_list_all: list = field(default_factory=list)
    lowest_list_all: list = field(default_factory=list)
    sliced_cost_list_all:list = field(default_factory=list)
    best_av_list: list = field(default_factory=list)
    noise:bool = None #noise simulation
    monte_carlo: bool = False #monte carlo
    mps: bool = None
    aws:bool = None
    target: str = None
    sigma:float = None
    best_av_to_date:str = None
    last_av:str = None
    
    def __post_init__(self):
        """This method is called after __init__"""
        self.subid = strftime('%H-%M-%S') # Generate subid if not provided
        super().__post_init__() # call parent's self init
        self.detailed_results_filename = self.find_detailed_results_filename()
        self.graph_filename = self.find_graph_filename()
        print(f'SubDataLogger instantiated.  Run ID = {self.runid} - {self.subid}')

    def calculate_parameter_numbers(self) -> int:
        """Calculate the number of parameters in a variational quantum circuit"""
        #num_params_per_qubit = MODE_DISPATCH[self.mode]['params_per_qubit']
        num_params_per_qubit = find_params_per_qubit(self.mode)
        #targets_sdk = MODE_DISPATCH[self.mode]['sdk']
        targets_sdk = find_sdk_from_dispatch_dir(self.mode)
        match targets_sdk:
            case 'aws':
                qubits_measured = find_qubits_measured(self.qubits, self.target)
                num_params =  num_params_per_qubit * qubits_measured * self.layers
            case 'qiskit':
                num_params = num_params_per_qubit * self.qubits * self.layers
            case _:
                raise Exception(f'Mode {self.mode} has not been coded for')
        return num_params

    def validate_input(self):
        """Validate the input fields"""
        #targets_sdk = MODE_DISPATCH[self.mode]['sdk']
        targets_sdk = find_sdk_from_dispatch_dir(self.mode)
        if not isinstance(self.quantum, bool):
            raise Exception(f'Input field quantum is not boolean')
        if not isinstance(self.hot_start, bool):
            raise Exception(f'Input field hot start is not boolean')
        if self.quantum:
            validate_gradient_type(self.gradient_type)
            allow_multiple_layers = find_multi_layers_allowed(self.mode)
            if self.formulation not in ['original', 'new']:
                raise Exception(f'Value {self.formulation} is not allowed for formulation' )
            if not isinstance(self.gray, bool):
                raise Exception(f'Input field gray is not boolean')
            if not isinstance(self.noise, bool):
                raise Exception(f'Input field noise is not boolean')
            if targets_sdk not in ['aws', 'qiskit']:
                raise Exception(f'mode = {self.mode} is not permitted for quantum')
            if not allow_multiple_layers and self.layers > 1:
                raise Exception(f'mode = {self.mode} is only for 1 layer')
            if self.mps and self.aws:
                raise Exception(f'MPS and AWS cannot both be true')
            if self.target not in TARGETS:
                raise Exception(f'Target {self.target} is not in TARGETS dictionary')
            if TARGETS[self.target]['type'] not in ['local_aws', 'aws'] and self.aws:
                raise Exception(f'AWS is set to true, but target {self.target} is not an AWS device')
            circuit_sdk = find_sdk_from_dispatch_dir(self.mode)
            targets_sdk = find_sdk(self.target)
            if circuit_sdk != targets_sdk:
                raise Exception(f'Mode {self.mode} is set up for {circuit_sdk}, but target {self.target} is set up for {targets_sdk}')
            if self.noise and targets_sdk != 'aws':
                raise Exception(f'Noise simulation is currently only not set up for AWS devices, and {self.target=}')
        else:
            if self.gradient_type not in ['SGD', 'SGD+X', 'Adam', 'Adam+X', 'RMSprop',]:
                raise Exception(f'Only certain gradient type are allowed for non quantum, not {self.gradient_type}')
            if targets_sdk != 'ml':
                raise Exception(f'mode = {self.mode} is not permitted for ml')
            if self.gradient_type in ['SGD+X', 'Adam+X'] and self.start:
                if self.hot_start:
                    raise Exception(f'Hot start is not allowed with SGD+X and AdamX')
            if self.mps is True:
                raise Exception(f'MPS simulator is only for quantum runs')
            if self.aws is True:
                raise Exception(f'AWS is only for quantum runs')
            if self.target != 'ml':
                raise Exception(f'{self.target=} and should be ml for non quantum runs')
    
    def save_results_to_csv(self):
        """Save the results to a CSV file"""
        results_dict = asdict(self)
        # delete items not needed in summary file
        del results_dict['index_list']
        del results_dict['average_list']
        del results_dict['lowest_list']
        del results_dict['sliced_list']
        del results_dict['average_list_all']
        del results_dict['lowest_list_all']
        del results_dict['sliced_cost_list_all']
        del results_dict['best_av_list']
        data_row = [results_dict]

        # Save the data to the specified CSV file
        file_path = self.summary_results_filename
        print(f"Saving data to {file_path}")
        if file_path.exists():
            try:
                with open(file_path, 'a', newline='') as csvfile:
                    fieldnames = data_row[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerows(data_row)

            except Exception as e:
                print(f"An error occurred while saving the data to {file_path}: {e}")    
        else:
            try:
                with open(file_path, 'w', newline='') as csvfile:
                    fieldnames = data_row[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    print('Writing header')
                    writer.writeheader()
                    print('Writing data')
                    writer.writerows(data_row)
                    print(f"Data saved to {file_path}")

            except Exception as e:
                print(f"An error occurred while saving the data to {file_path}: {e}")

    def update_constants_from_dict(self, 
                                   data_dict: dict,
                                   ):
        """Update the constants from a dictionary"""
        self.quantum = format_boolean(data_dict['quantum'])
        self.locations = int(data_dict['locations'])
        self.shots = int(data_dict['shots'])
        self.iterations = int(data_dict['iterations'])
        self.gray = format_boolean(data_dict['gray'])
        self.hot_start = format_boolean(data_dict['hot_start'])
        self.gradient_type = data_dict['gradient_type']
        self.formulation = data_dict['formulation']
        self.cache_max_size = CACHE_MAX_SIZE
        self.mode = int(data_dict['mode'])
        self.layers = int(data_dict['layers'])
        self.target = data_dict['target']
        if not self.quantum:
            self.std_dev = float(data_dict['std_dev'])
            self.lr = float(data_dict['lr'])
            self.weight_decay = float(data_dict['weight_decay'])
            if self.gradient_type == 'SGD':
                self.momentum = float(data_dict['momentum'])
        if self.quantum:
            self.slice = float(data_dict['slice'])
            self.alpha = float(data_dict['alpha'])
            self.big_a = float(data_dict['big_a'])
            self.c = float(data_dict['c'])
            self.gamma = float(data_dict['gamma'])
            self.eta = float(data_dict['eta'])
            self.s = float(data_dict['s'])
            self.noise = format_boolean(data_dict['noise'])
            self.mps = format_boolean(data_dict['mps'])
            self.aws = format_boolean(data_dict['aws'])
            self.target = data_dict['target']
            
    def update_general_constants_from_config(self):
        """Update general constants from the config file"""
        self.locations = LOCATIONS
        self.shots = SHOTS
        self.mode = MODE
        self.iterations = ITERATIONS
        self.gray = GRAY
        self.hot_start = HOT_START
        self.gradient_type = GRADIENT_TYPE
        self.formulation = DECODING_FORMULATION
        self.layers= NUM_LAYERS
        self.cache_max_size = CACHE_MAX_SIZE
        self.target = TARGET

    def update_quantum_constants_from_config(self):
        """Update constants needed for quantum from config file"""
        self.quantum = True
        self.alpha = ALPHA
        self.big_a = BIG_A
        self.c = C
        self.eta = ETA
        self.gamma = GAMMA
        self.s = S
        self.noise = SIMULATE_NOISE
        self.mps = MPS
        self.aws = AWS

    def update_ml_constants_from_config(self):
        """Update constants needed for ML from config file"""
        self.quantum = False
        self.noise = False # noise is for quantum, not classical
        self.std_dev = STD_DEV
        self.lr = LR
        self.momentum = MOMENTUM
        self.weight_decay = WEIGHT_DECAY

    def update_cache_statistics(self, 
                                cost_fn: Callable,
                                ):
        """Update cache statistics"""
        if hasattr(cost_fn, 'report_cache_stats'):
            items, hits, misses = cost_fn.report_cache_stats()
            self.cache_items = items
            self.cache_hits = hits
            self.cache_misses = misses
        else:
            self.cache_items = 0
            self.cache_hits = 0
            self.cache_misses = 0

        if hasattr(cost_fn, 'clear_cache'):
            cost_fn.clear_cache() 

    def find_detailed_results_filename(self):
        """Create the filepath for the detailed results"""
        detailed_results_filename = Path.joinpath(self.results_sub_path, f'{self.subid}.csv')
        return detailed_results_filename
    
    def find_graph_filename(self):
        """Create the filepath for the graphs results"""
        graph_filename = Path.joinpath(self.graph_sub_path, f'{self.subid}.png')
        return graph_filename
    
    def save_detailed_results(self):
        """Save detailed data"""
        file_path = self.detailed_results_filename
        index_list = self.index_list
        average_list = list(map(float, self.average_list))
        lowest_list = list(map(float, self.lowest_list))
        if self.sliced_list != [] and self.best_av_list != []:
            raise Exception(f'Cannot write to both sliced list and best_av')
        if self.sliced_list != []:
            sliced_list = list(map(float, self.sliced_list))
            field_name_list = ['index_list', 'average_list', 'lowest_list', 'sliced_list']
        if self.best_av_list != []:
            best_av_list = list(map(float, self.best_av_list))
            field_name_list = ['index_list', 'average_list', 'lowest_list', 'best_av_list']
        else:
            field_name_list = ['index_list', 'average_list', 'lowest_list']
        try:
            with open(file_path, mode="a", newline="") as file: 
                writer = csv.writer(file) 
                writer.writerow(field_name_list)
                if self.sliced_list != [] and self.best_av_list != []:
                    raise Exception(f'Cannot write to both sliced list and best_av')
                if self.sliced_list != []:
                    for row in zip(index_list, average_list, lowest_list, sliced_list):
                        writer.writerow(row)
                if self.best_av_list != []:
                    for row in zip(index_list, average_list, lowest_list, best_av_list):
                        writer.writerow(row)
                else:
                    for row in zip(index_list, average_list, lowest_list):
                        writer.writerow(row)
                print(f'Detailed data for Run ID: {self.runid} - {self.subid} successfully added to {file_path}')

        except Exception as e:
                print(f"An error occurred while saving the data to {file_path}: {e}")

    def save_plot(self):
        """Plot results"""
        title = f'Evolution of loss for Run ID {self.runid} - {self.subid}' 

        print(f'Graph for Run ID: {self.runid}-{self.subid} being saved to {self.graph_filename}')
        
        cost_graph_multi(filename=self.graph_filename,
                         x_list=self.index_list,
                         av_list=self.average_list_all,
                         lowest_list=self.lowest_list_all,
                         sliced_list=self.sliced_cost_list_all,
                         main_title=title,
                         best=self.best_dist
                         )