import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2

from ._pipe_data import PipeData
from ._path_to_csv import PathToCsv

__all__ = ["PipeDataCsvWorker"]

class PipeDataCsvWorker(PipeData):
    """A class for read data from csv files and research it"""
    
    # The path to a data file like "*_data.csv"
    path_to_data_file = PathToCsv()
    # The path to a data file like "*_defects.csv"
    path_to_defects_file = PathToCsv()
    # The path to a data file like "*_pipe.csv"
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

    def write_to_db(self, **kwargs):
        """
        Write data to postgres database 
        """

        # sql insert query template
        INSERT_INPUT_DATA_CELL_QUERY = """
            INSERT INTO data_cell(
            file_id, 
            row_id, 
            detector_id, 
            time_values, 
            amplitude_values,
            defect_depth) VALUES (%s,%s,%s,%s,%s,%s);
            """

        conn = psycopg2.connect(**kwargs)
        
        last_file_id = 0
            
        with conn.cursor() as cursor:
            cursor.execute('SELECT MAX(file_id) FROM data_cell')
            result = cursor.fetchone()
            if result[0] is not None: 
                last_file_id = result[0]+1    

        with conn.cursor() as cursor:
            # insert data in data_cell table
            for i in range(self._data_df.shape[0]):
                for j in range(self._data_df.shape[1]):
                        cursor.execute(INSERT_INPUT_DATA_CELL_QUERY,
                                        (last_file_id, # file id
                                        i, # row id
                                        j, # col id
                                        list(self._data_df.iloc[i,j][:32]), # time_values list
                                        list(self._data_df.iloc[i,j][32:]), # amplitude_values list
                                        self._defects_df.iloc[i,j])) # defect_depth
        conn.commit()
        conn.close()
                
    
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
            
            time_vals = num_pars[:,0]
            amp_vals = num_pars[:,1]

            time_vals = np.pad(time_vals, (0, abs(time_vals.size-32)), constant_values=(0))
            amp_vals = np.pad(amp_vals, (0, abs(amp_vals.size-32)), constant_values=(0))
            
            return np.concatenate((time_vals, amp_vals), axis=0)
    
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
        self._data_df = data_df
        self._extend = {'left':0,'top':0,'right':data_df.shape[1], 'bottom':data_df.shape[0]}