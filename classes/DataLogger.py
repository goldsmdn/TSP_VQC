#Class to handle data logging
from time import strftime
from pathlib import Path
import csv
from modules.config import (RESULTS_DIR, 
                            RESULTS_FILE,
                            GRAPH_DIR
                            )

from modules.graph_functions import cost_graph_multi
                           
class DataLogger:
    """Parent - also responsible for creating graph folders if needed"""
    def __init__(self):
        self.runid = strftime('%Y%m%d-%H-%M-%S')
        print(f'Data logger instantiated.  Run ID: {self.runid}')
        self.header_written = True
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
        graph_path = Path(GRAPH_DIR)
        self.graph_sub_path = Path.joinpath(graph_path, self.runid)
        self.graph_sub_path.mkdir(parents=True, exist_ok=True)
        print(f'Graphs are saved in folder {self.graph_sub_path}')

    def create_results_path(self):
        """Create a folder for results"""
        results_path = Path(RESULTS_DIR)
        self.detailed_results_sub_path = Path.joinpath(results_path, self.runid)
        self.detailed_results_sub_path.mkdir(parents=True, exist_ok=True)
        print(f'Detailed results are saved in folder {self.detailed_results_sub_path}')

class SubDataLogger(DataLogger):
    """Child - responsible for writing data to csv for each run"""
    def __init__(self, parent: DataLogger):
        super().__init__()
        self.runid = parent.runid
        self.parent = parent
        self.data_sub_path = Path(RESULTS_DIR)
        self.create_results_path()
        self.create_graph_path()
        self.subid = strftime('%H-%M-%S')
        self.summary_results_filename = Path.joinpath(self.data_sub_path, RESULTS_FILE)
        self.detailed_results_filename = Path.joinpath(self.detailed_results_sub_path,f'{self.subid}.csv')
        self.graph_filename = Path.joinpath(self.graph_sub_path,f'{self.subid}.png')
        self.full_id = f'{self.runid} - {self.subid}'

        print(f'SubDataLogger instantiated.  Run ID = {self.runid} - {self.subid}')

    def save_dict_to_csv(self, data: dict):
        """Save data to csv"""
        with open(self.summary_results_filename, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if not(self.parent.header_written):
                writer.writeheader()
                self.parent.header_written = True
            writer.writerow(data)
            print(f'Summary data for Run ID: {self.full_id} successfully added to {self.summary_results_filename}')

    def save_detailed_results(self, 
                              index_list:list,
                              average_list:list,
                              lowest_list:list,
                              sliced_list:list=None
                              ):
        """Save detailed data"""

        average_list = list(map(float, average_list))
        lowest_list = list(map(float, lowest_list))
        if sliced_list:
            sliced_list = list(map(float, sliced_list))

        with open(self.detailed_results_filename, mode="a", newline="") as file:    
            writer = csv.writer(file)
            # write header
            writer.writerow(self.results_field_names)
            #write rows
            if sliced_list:
                for row in zip(index_list, average_list, lowest_list, sliced_list):
                    writer.writerow(row)
            else:
                for row in zip(index_list, average_list, lowest_list):
                    writer.writerow(row)
            print(f'Detailed data for Run ID: {self.runid} - {self.subid} successfully added to {self.detailed_results_filename}')

    def save_plot(self, 
                  index_list:list,
                  av_cost_list_all:list,
                  lowest_list_all:list,
                  sliced_cost_list_all:list=None,
                  best_dist:float=None,
                  ):
        """plot results"""
        title = f'Evolution of loss for Run ID {self.runid} - {self.subid}' 
        
        cost_graph_multi(self.graph_filename,
                         x_list=index_list,
                         av_list=av_cost_list_all,
                         lowest_list=lowest_list_all,
                         sliced_list=sliced_cost_list_all,
                         main_title=title,
                         best=best_dist
                         )
        print(f'Graph for Run ID: {self.runid}-{self.subid} saved to {self.graph_filename}')