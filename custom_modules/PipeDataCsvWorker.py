import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('custom_modules')
from PathToCsv import PathToCsv

# A class for read data from csv files
# and research it
class PipeDataCsvWorker:

    path_to_data_file = PathToCsv()
    path_to_defects_file = PathToCsv()
    path_to_pipe_file = PathToCsv()
    
    def __init__(self, 
                 path_to_data_file,
                 path_to_defects_file,
                 path_to_pipe_file):
        self.path_to_data_file = path_to_data_file
        self.path_to_defects_file = path_to_defects_file
        self.path_to_pipe_file = path_to_pipe_file
        self.read_csv_data()

    def get_defects_df(self):
        return self._defects_df

    def get_data_df(self):
        return self._data_df
    
    # roll data_df and defects_df elements along a given axis
    def roll_dfs_along_axis(self, shift, axis=0):
        # roll any dataframe like numpy.roll method
        def roll_df(df, shift, axis):
            df_values = np.roll(df.to_numpy(), shift, axis)
            df_index = (df.index.to_numpy() if axis == 1 
                            else np.roll(df.index.to_numpy(), shift))
            df_columns = (df.columns.to_numpy() if axis == 0 
                              else np.roll(df.columns.to_numpy(), shift))
        
            return pd.DataFrame(data=df_values, index=df_index, 
                                columns=df_columns)
        
        if not type(shift) is int:
            raise TypeError("A shift should be int")
        if not type(axis) is int:
            raise TypeError("An axis should be int")
        if not axis in [0, 1]:
            raise ValueError('An axis should be 0 or 1')

        self._data_df = roll_df(self._data_df, shift, axis)
        self._defects_df = roll_df(self._defects_df, shift, axis)
        

    # draw a defects map from defects_df data
    def draw_defects_map(self, 
                         title='Развернутая карта дефектов',
                         xlabel: str = 'Номер датчика', 
                         ylabel: str = 'Номер измерения',
                         x_ticks_step = 50,
                         y_ticks_step = 20):
        
        if not type(title) is str:
            raise TypeError("A title should be str")
        if not type(xlabel) is str:
            raise TypeError("A xlabel should be str")
        if not type(ylabel) is str:
            raise TypeError("A ylabel should be str")

        # add fillers between itemns of an array
        # only in case when prev value less than
        # next one
        def add_fillers(arr, i=0, step=1):
            if i > len(arr)-2:
                return arr
            if abs(arr[i] - arr[i+1]) >= step * 1.5:
                if arr[i] < arr[i+1]:
                    arr.insert(i+1, arr[i]+step)
            return add_fillers(arr, i+1, step)

        # merge biggest and smalles value
        # in one like 'big/small'
        # for roll operation only
        def merge_corners(arr):
            for i in range(len(arr)):
                if arr[i] == max(arr):
                    arr[i] = f'{max(arr)}/{min(arr)}'
                    del arr[i+1]
                    break
            return arr
        
        with plt.style.context('dark_background'):

            fig, ax = plt.subplots()
            
            # decorate     
            fig.set_figwidth(18)
            fig.set_figheight(8)
            fig.patch.set_alpha(0.0)
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_title(title, fontsize=25) 
            ax.set_xlabel(xlabel, fontsize=20) 
            ax.set_ylabel(ylabel, fontsize=20) 
            ax.tick_params(axis='both', which='both', labelsize = 20)
            
            df = self._defects_df

            # get columns and indexes list of int type
            cols = [int(col.split('_')[1]) for col in df.columns.values]
            indexes = df.index.values.tolist()

            # array with essential x labels items
            new_x_labels = [cols[0], cols[-1]]
            new_y_labels = [indexes[0], indexes[-1]]

            # add essential items if dataframe was rolled
            if max(cols) != cols[-1]:
                new_x_labels.insert(1, max(cols))
                new_x_labels.insert(2, min(cols))

            # add essential items if dataframe was rolled
            if max(indexes) != indexes[-1]:
                new_y_labels.insert(1, max(indexes))
                new_y_labels.insert(2, min(indexes))

            # add fillers to labels lists
            new_x_labels = add_fillers(new_x_labels,step=x_ticks_step)
            new_y_labels = add_fillers(new_y_labels,step=y_ticks_step)

            # merge border values together if
            # dataframe was rolled
            if max(cols) != cols[-1]:
                new_x_labels = merge_corners(new_x_labels)
            if max(indexes) != indexes[-1]:    
                new_y_labels = merge_corners(new_y_labels)

            # make all items str type
            new_x_labels = [str(item) for item in new_x_labels]
            new_y_labels = [str(item) for item in new_y_labels]

            # calculate locks of label values
            new_x_locs = [cols.index(int(re.match("[0-9]+", item)[0])) 
                          for item in new_x_labels]
            new_y_locs = [indexes.index(int(re.match("[0-9]+", item)[0])) 
                          for item in new_y_labels]
            
            ax.pcolormesh(df)

            plt.xticks(new_x_locs, new_x_labels)
            plt.yticks(new_y_locs, new_y_labels)
            

        plt.show()
    
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

    def read_csv_data(self):
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