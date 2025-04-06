#Class to handle data logging
from time import strftime
from pathlib import Path
import csv
from dataclasses import dataclass, asdict, field

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
                            WEIGHT_DECAY
                            )

from modules.graph_functions import cost_graph_multi
from modules.helper_functions_tsp import (validate_gradient_type,
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
        #print(f'Data logger instantiated.  Run ID: {self.runid}')

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

    def __post_init__(self):
        """This method is called after __init__"""
        self.subid = strftime('%H-%M-%S') # Generate subid if not provided
        super().__post_init__() # call parent's self init
        self.detailed_results_filename = self.find_detailed_results_filename()
        self.graph_filename = self.find_graph_filename()
        print(f'SubDataLogger instantiated.  Run ID = {self.runid} - {self.subid}')

    def validate_input(self):
        """Validate the input fields"""
        if not isinstance(self.quantum, bool):
            raise Exception(f'Input field quantum is not boolean')
        if not isinstance(self.gray, bool):
            raise Exception(f'Input field gray is not boolean')
        if not isinstance(self.hot_start, bool):
            raise Exception(f'Input field hot start is not boolean')
        if self.formulation not in ['original', 'new']:
            raise Exception(f'Value {self.formulation} is not allowed for formulation' )
        if self.quantum:
            validate_gradient_type(self.gradient_type)
            if self.mode not in [1,2]:
                raise Exception(f'mode = {self.mode} is not permitted for quantum')
        else:
            if self.gradient_type not in ['SGD', 'Adam', 'RMSprop']:
                raise Exception(f'Only gradient type SGD is allowed for non quantum, not {self.gradient_type}')
    
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
        data_row = [results_dict]

        # Save the data to the specified CSV file
        file_path = self.summary_results_filename
        print(f"Saving data to {file_path}")
        if file_path.exists():
            try:
                with open(file_path, 'a', newline='') as csvfile:
                    fieldnames = data_row[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    #print('Writing data')
                    writer.writerows(data_row)
                    #print(f"Data saved to {file_path}")

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

    def update_constants_from_dict(self, data_dict):
        """Update the constants from a dictionary"""
        self.quantum = format_boolean(data_dict['quantum'])
        self.locations = int(data_dict['locations'])
        self.slice = float(data_dict['slice'])
        self.shots = int(data_dict['shots'])
        self.iterations = int(data_dict['iterations'])
        self.gray = format_boolean(data_dict['gray'])
        self.hot_start = format_boolean(data_dict['hot_start'])
        self.gradient_type = data_dict['gradient_type']
        self.formulation = data_dict['formulation']
        self.cache_max_size = CACHE_MAX_SIZE
        if not self.quantum:
            self.layers = int(data_dict['layers'])
            self.std_dev = float(data_dict['std_dev'])
            self.lr = float(data_dict['lr'])
            self.weight_decay = float(data_dict['weight_decay'])
            self.momentum = float(data_dict['momentum'])
        if self.quantum:
            self.mode = int(data_dict['mode'])
            self.alpha = float(data_dict['alpha'])
            self.big_a = float(data_dict['big_a'])
            self.c = float(data_dict['c'])
            self.gamma = float(data_dict['gamma'])
            self.eta = float(data_dict['eta'])
            self.s = float(data_dict['s'])
        
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
        self.cache_max_size = CACHE_MAX_SIZE

    def update_quantum_constants_from_config(self):
        """Update constants needed for quantum from config file"""
        self.alpha = ALPHA
        self.big_a = BIG_A
        self.c = C
        self.eta = ETA
        self.gamma = GAMMA
        self.s = S
        self.cache_max_size = CACHE_MAX_SIZE

    def update_ml_constants_from_config(self):
        """Update constants needed for ML from config file"""
        self.layers= NUM_LAYERS
        self.std_dev = STD_DEV
        self.lr = LR
        self.momentum = MOMENTUM
        self.weight_decay = WEIGHT_DECAY


    def update_cache_statistics(self, cost_fn):
        """update cache statistics"""
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
        field_name_list = ['index_list', 'average_list', 'lowest_list', 'sliced_list']
        file_path = self.detailed_results_filename
        index_list = self.index_list
        average_list = list(map(float, self.average_list))
        lowest_list = list(map(float, self.lowest_list))
        if self.sliced_list != []:
            sliced_list = list(map(float, self.sliced_list))
        try:
            with open(file_path, mode="a", newline="") as file: 
                writer = csv.writer(file) 
                writer.writerow(field_name_list)
                if self.sliced_list != []:
                    for row in zip(index_list, average_list, lowest_list, sliced_list):
                        writer.writerow(row)
                else:
                    for row in zip(index_list, average_list, lowest_list):
                        writer.writerow(row)
                print(f'Detailed data for Run ID: {self.runid} - {self.subid} successfully added to {file_path}')

        except Exception as e:
                print(f"An error occurred while saving the data to {file_path}: {e}")

    def save_plot(self):
        """plot results"""
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