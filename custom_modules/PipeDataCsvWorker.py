import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('custom_modules')
from PipeData import PipeData
from PathToCsv import PathToCsv

class PipeDataCsvWorker(PipeData):
    """A class for read data from csv files and research it"""

    path_to_data_file = PathToCsv()
    path_to_defects_file = PathToCsv()
    path_to_pipe_file = PathToCsv()
    
    def __init__(self, 
                 path_to_data_file: str,
                 path_to_defects_file: str,
                 path_to_pipe_file: str):
        self.path_to_data_file = path_to_data_file
        self.path_to_defects_file = path_to_defects_file
        self.path_to_pipe_file = path_to_pipe_file
        self._shift = [0,0]
        self.read_data()
    
    def _get_df_from_defects_file(self):
        """Read data file like "*_defects.csv" and returns
           pandas dataframe with preprocessed data from it"""
        using_columns = ['row_min', 'row_max', 'detector_min', 
                         'detector_max', 'fea_depth']
        df = pd.read_csv(self.path_to_defects_file, delimiter=';')
        return df[using_columns]
        
    def _get_df_from_pipe_file(self):
        """Read data file like "*_pipe.csv" and returns 
           pandas dataframe"""
        df = pd.read_csv(self.path_to_pipe_file, delimiter=';')
        return df

    def _get_df_from_data_file(self):
        """Read data file like "*_data.csv" and returns
           pandas dataframe with preprocessed data from it"""
        def split_every_cell_string_value_to_numpy_array_of_64_values(df_cell_value):
            """Converte all data cells values from given pandas dataframe from
               string (describes 2D values array) to 1D float numpy array of 64 items"""
            num_pars = re.findall(r'(-?[0-9]+(\.[0-9]+)*):(-?[0-9]+(\.[0-9]+)*)', df_cell_value)
            num_pars = np.array([[item[0], item[2]] for item in num_pars]).astype(float)
            if num_pars.size == 0:
                return np.zeros((64))
            num_pars = np.pad(num_pars, ((0,32 % num_pars.shape[0]),(0,0)), constant_values=(0))
            
            return np.concatenate((num_pars[:,0], num_pars[:,1]), axis=0)
    
        df = pd.read_csv(self.path_to_data_file, delimiter=';').astype(str)
        df = df.drop(['position'], axis=1)
        df = df.set_index('row')
        df = df.map(split_every_cell_string_value_to_numpy_array_of_64_values)
        return df

    def read_data(self):
        """Read data from set of .csv files"""
        data_df = self._get_df_from_data_file()
        defects_df = self._get_df_from_defects_file()
        pipe_df = self._get_df_from_pipe_file()
        
        # create defects depths mask
        # create base zeros dataframe with size like data_df
        base_df = pd.DataFrame(data = 0.0, index = data_df.index,
                               columns = data_df.columns)
        
        # read line-by-line defects_df
        # get defects location and mark by ones
        for row_name in defects_df.index.values.tolist():
            (row_min, row_max,
             detector_min, detector_max,
             fea_depth ) = defects_df.astype('object').loc[row_name].to_list()
            
            # mark defect location in base dataframe
            if (detector_min < detector_max):
                base_df.iloc[row_min:row_max+1,detector_min:detector_max+1] = fea_depth
                continue
        
            base_df.iloc[row_min:row_max+1,detector_min:data_df.shape[1]] = fea_depth
            base_df.iloc[row_min:row_max+1,:detector_max+1] = fea_depth

        defects_df = base_df
        self._defects_df = defects_df
        self._data_df = data_df        self._orig_items_mask = np.full(defects_df.shape, True)