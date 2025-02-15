#Class to handle data logging
from time import strftime
from pathlib import Path

class DataLogger:
    def __init__(self):
        self.runid = strftime('%Y%m%d-%H-%M-%S')
        print(f'Data logger instantiated.  Run ID: {self.runid}')

        graph_path = Path('graphs')
        self.graph_sub_path = Path.joinpath(graph_path, self.runid)
        self.graph_sub_path.mkdir(parents=True, exist_ok=True)
        print(f'Folder graph_sub_path = {self.graph_sub_path} is created for graphs')

        data_path = Path('data')
        self.data_sub_path = Path.joinpath(data_path, self.runid)
        self.data_sub_path.mkdir(parents=True, exist_ok=True)
        print(f'Folder data_sub_path = {self.data_sub_path} is created for data')



