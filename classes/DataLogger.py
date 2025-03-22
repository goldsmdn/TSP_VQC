#Class to handle data logging
from time import strftime
from pathlib import Path
import csv
from modules.config import (RESULTS_DIR, 
                            RESULTS_FILE,
                            GRAPH_DIR
                            )
                           
class DataLogger:
    """Parent - also responsible for creating graph folders if needed"""
    def __init__(self):
        self.runid = strftime('%Y%m%d-%H-%M-%S')
        print(f'Data logger instantiated.  Run ID: {self.runid}')
        self.header_written = True
        #self.data_header_written = False
        self.fieldnames = ['runid', 
                           'subid', 
                           'quantum', 
                           'locations', 
                           'slice', 
                           'shots', 
                           'layers', 
                           'std_dev', 
                           'mode', 
                           'iterations', 
                           'gray', 
                           'hot_start', 
                           'gradient_type', 
                           'formulation',
                           'lr', 
                           'weight_decay', 
                           'momentum',
                           'alpha',
                           'big_a', 
                           'c', 
                           'eta', 
                           'gamma', 
                           's', 
                           'print_frequency',
                           'qubits',
                           'elapsed', 
                           'hot_start_dist', 
                           'best_dist_found', 
                           'best_dist',
                           'iteration_found',
                           'cache_max_size', 
                           'cache_items', 
                           'cache_hits', 
                           'cache_misses', 
                          ]
        self.results_field_names = ('epoch',
                                    'av_cost',
                                    'lowest_cost',
                                    'sliced_cost'
                                    )

    def create_graph_path(self):
        """Create a folder for graphs"""
        #graph_path = Path('graphs')
        graph_path = Path(GRAPH_DIR)
        self.graph_sub_path = Path.joinpath(graph_path, self.runid)
        self.graph_sub_path.mkdir(parents=True, exist_ok=True)
        print(f'Folder {self.graph_sub_path} is created for graphs')

    def create_results_path(self):
        """Create a folder for results"""
        results_path = Path(RESULTS_DIR)
        self.detailed_results_sub_path = Path.joinpath(results_path, self.runid)
        self.detailed_results_sub_path.mkdir(parents=True, exist_ok=True)
        print(f'Folder {self.detailed_results_sub_path} is created for detailed results')

class SubDataLogger(DataLogger):
    """Child - responsible for writing data to csv for each run"""
    def __init__(self, parent: DataLogger):
        super().__init__()
        self.runid = parent.runid
        self.parent = parent
        self.data_sub_path = Path(RESULTS_DIR)
        self.create_results_path()
        self.subid = strftime('%H-%M-%S')
        self.summary_results_filename = Path.joinpath(self.data_sub_path, RESULTS_FILE)
        self.detailed_results_filename = Path.joinpath(self.detailed_results_sub_path,f'{self.subid}.csv')
        self.full_id = f'{self.runid} - {self.subid}'

        print(f'SubDataLogger instantiated.  Run ID = {self.runid} - {self.subid}')
        print(f'Results to be written to the {self.data_sub_path} folder')

    def save_dict_to_csv(self, data: dict):
        """Save data to csv"""
        with open(self.summary_results_filename, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if not(self.parent.header_written):
                writer.writeheader()
                self.parent.header_written = True
            writer.writerow(data)
            print(f'Data for Run ID: {self.full_id} successfully added to {self.summary_results_filename}')

    def save_detailed_results(self, 
                              epoch:list,
                              av_cost:list,
                              lowest_cost:list,
                              sliced_cost:list=None
                              ):
        """Save detailed data"""
        
        #rows = zip(epoch, av_cost, lowest_cost, sliced_cost)
        #rows = zip(epoch, map(float, av_cost), map(float, lowest_cost), map(float, sliced_cost))
        #rows = list(zip(epoch, map(float, av_cost), map(float, lowest_cost), map(float, sliced_cost)))
        av_cost = list(map(float, av_cost))
        lowest_cost = list(map(float, lowest_cost))
        sliced_cost = list(map(float, sliced_cost))

        with open(self.detailed_results_filename, mode="a", newline="") as file:    
            writer = csv.writer(file)
            # write header
            writer.writerow(self.results_field_names)
            #write rows
            for row in zip(epoch, av_cost, lowest_cost, sliced_cost):
                writer.writerow(row)
            #writer.writerow(rows)
            print(f'Data for Run ID: {self.runid} - {self.subid} successfully added to {self.detailed_results_filename}')


    