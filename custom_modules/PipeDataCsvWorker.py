import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('custom_modules')
from PipeData import PipeData
from PathToCsv import PathToCsv

# A class for read data from csv files
# and research it
class PipeDataCsvWorker(PipeData):

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
        self.read_data()
    
    # Read data file like "*_defects.csv" and returns 
    # pandas dataframe with preprocessed data from it    
    def _get_df_from_defects_file(self):
        using_columns = ['row_min', 'row_max', 'detector_min', 
                         'detector_max', 'fea_depth']
        df = pd.read_csv(self.path_to_defects_file, delimiter=';')
        return df[using_columns]
        
    # Read data file like "*_pipe.csv" and returns 
    # pandas dataframe   
    def _get_df_from_pipe_file(self):
        df = pd.read_csv(self.path_to_pipe_file, delimiter=';')
        return df

    # Read data file like "*_data.csv" and returns 
    # pandas dataframe with preprocessed data from it
    def _get_df_from_data_file(self):
        # Converte all data cells values from given pandas 
        # dataframe from string (describes 2D values array)
        # to 1D float numpy array of 64 items
        def split_every_cell_string_value_to_numpy_array_of_64_values(df_cell_value):
            # Divide each pair of numbers to individual
            # numbers and collects them to a list
            def split_numbers_pairs_string_list_to_list(pairs_list):
                result_list = list()
                for numbers_pair in pairs_list:
                    temp = str(numbers_pair).split(':')
                    # Adds new numbers to result list 
                    # only if the number has a pair
                    if len(temp) == 2:
                        result_list.append(temp)
                return result_list
    
            # Check if init string does not have
            # numbers returns 64 zeros numpy array
            if not bool(re.search(r'\d', df_cell_value)):
                return np.zeros(64).astype(float)
    
            # Create 1D list of strings where each
            # one is a pair of numbers
            pairs_list = str(df_cell_value).split(',')
    
            # Create 2D list of strings where 1
            # dimension is a quantity of the pairs,
            # 2 dimension stores numbers of a pair
            # separately as string
            pairs_list = split_numbers_pairs_string_list_to_list(pairs_list)
            result_array = np.array(pairs_list).astype(float)
            
            # Add number placeholders to the array so it
            # consists from 64 items
            result_array = np.pad(result_array, 
                                  ((0,32 - result_array.shape[0]),(0,0)), 
                                  'constant', constant_values=(0))
            
            # Change numbers order in the array. 
            # It was like: time_1,amplitude_1,time_2,amplitude_2,....
            # Now it like: time_1,...,time_32,amplitude_1,...,amplitude_32
            return np.concatenate((result_array[:,0], result_array[:,1]), axis=0)
    
        df = pd.read_csv(self.path_to_data_file, delimiter=';')
        df = df.drop(['position'], axis=1)
        df = df.set_index('row')
        df = df.map(split_every_cell_string_value_to_numpy_array_of_64_values)
        return df

    def read_data(self):
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
        self._data_df = data_df