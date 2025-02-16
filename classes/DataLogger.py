#Class to handle data logging
from time import strftime
from pathlib import Path
import csv

class DataLogger:
    """Parent - also responsible for creating graph folders if needed"""
    def __init__(self):
        self.runid = strftime('%Y%m%d-%H-%M-%S')
        print(f'Data logger instantiated.  Run ID: {self.runid}')
        self.header_written = False
        self.fieldnames = ['runid', 'subid', 'locations', 'slice', 'shots', 'mode', 
                    'iterations', 'gray', 'hot_start', 'gradient_type', 'formulation','alpha',
                    'big_a', 'c', 'eta', 'gamma', 's', 'print_frequency',
                    'cache_max_size', 'elapsed', 
                    'hot_start_dist', 'hot_start_cost', 'best_dist_found', 
                    'best_dist','iteration_found',
                    'cache_items', 'cache_hits', 'cache_misses', 
                    ]

    def create_graph_path(self):
        graph_path = Path('graphs')
        self.graph_sub_path = Path.joinpath(graph_path, self.runid)
        self.graph_sub_path.mkdir(parents=True, exist_ok=True)
        print(f'Folder graph_sub_path = {self.graph_sub_path} is created for graphs')

class SubDataLogger(DataLogger):
    def __init__(self, parent: DataLogger):
        super().__init__()
        self.runid = parent.runid
        self.parent = parent

        data_path = Path('data')
        self.data_sub_path = Path.joinpath(data_path, self.runid)
        self.data_sub_path.mkdir(parents=True, exist_ok=True)

        self.subid = strftime('%H-%M-%S')
        self.filename = Path.joinpath(self.data_sub_path, f'{self.runid}.csv')
        self.full_id = f'{self.runid} - {self.subid}'

        print(f'SubDataLogger instantiated.  Run ID = {self.runid} - {self.subid}')
        print(f'Folder data_sub_path = {self.data_sub_path} is used for data writing')
        print(f'Data will be added to file = {self.filename}')

    def save_dict_to_csv(self, data: dict):
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            #print(f'self.header_written = {self.header_written}')
            if not(self.parent.header_written):
                #print(f'Writing header. self.header_written = {self.header_written}')
                writer.writeheader()
                self.parent.header_written = True
                #print(f'Writen header. self.header_written = {self.header_written}')
            writer.writerow(data)
            #print(f'Writing row. self.header_written = {self.header_written}')
            #print(f'Data_dict = {data}')
            print(f'Data for Run ID: {self.full_id} successfully added to {self.filename}')