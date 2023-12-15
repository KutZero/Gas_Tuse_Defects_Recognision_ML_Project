# Extensions importing
import pandas as pd
import numpy as np
import re

# Data reading ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Reads data file like "*_data.csv" and returns 
# pandas dataframe with preprocessed data from it
def get_df_from_data_file(path: str):
    # Convertes all data cells values from given pandas 
    # dataframe from string (describes 2D values array)
    # to 1D float numpy array of 64 items
    def split_every_cell_string_value_to_numpy_array_of_64_values(df_cell_value):
        # Divides each pair of numbers to individual
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

        # Checks if init string does not have
        # numbers returns 64 zeros numpy array
        if not bool(re.search(r'\d', df_cell_value)):
            return np.zeros(64).astype(float)

        # Creates 1D list of strings where each
        # one is a pair of numbers
        pairs_list = str(df_cell_value).split(',')

        # Creates 2D list of strings where 1
        # dimension is a quantity of the pairs,
        # 2 dimension stores numbers of a pair
        # separately as string
        pairs_list = split_numbers_pairs_string_list_to_list(pairs_list)

        result_array = np.array(pairs_list).astype(float)
        
        # Add number placeholders to the array so it
        # consists from 64 items
        result_array = np.pad(result_array, ((0,32 - result_array.shape[0]),(0,0)), 'constant', constant_values=(0))
        
        # Changes numbers order in the array. 
        # It was like:
        # time_1,amplitude_1,time_2,amplitude_2,....
        # Now it like:
        # time_1,...,time_32,amplitude_1,...,amplitude_32
        time_values = result_array[:,0]
        amplitude_values = result_array[:,1]
        
        result_array = np.concatenate((time_values, amplitude_values), axis=0)
        
        return result_array

    df = pd.read_csv(path, delimiter=';')
    df = df.drop(['position'], axis=1)
    df['row'] = df['row'].astype(int)
    df = df.set_index('row')
    df = df.map(split_every_cell_string_value_to_numpy_array_of_64_values)
    return df
    
# Reads data file like "*_defects.csv" and returns 
# pandas dataframe with preprocessed data from it    
def get_df_from_defects_file(path: str):
    df = pd.read_csv(path, delimiter=';')
    df = pd.concat([df.loc[:,'row_min':'detector_max'],
                    df.loc[:,'fea_length':'fea_depth']],
                    axis=1)
    return df
    
# Reads data file like "*_pipe.csv" and returns 
# pandas dataframe   
def get_df_from_pipe_file(path: str):
    df = pd.read_csv(path, delimiter=';')
    return df
