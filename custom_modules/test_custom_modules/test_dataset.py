from custom_modules.dataset import (
    get_batch_generator,
    get_crop_generator,
    get_augmented_crop_generator,
    get_x_and_y_data_dfs,
    _get_df_from_defects_file,
    _split_cell_string_value_to_numpy_array_of_64_values,
    _get_df_from_data_file)

import pytest
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import itertools

from contextlib import nullcontext as does_not_raise
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field

@pytest.fixture
def csv_file_path(tmpdir_factory):
    df = pd.DataFrame({'col1': [1, 2, 1], 
                       'col2': [3, 4, 1],
                       'col3': [5, 6, 1]}) 
    filename = str(tmpdir_factory.mktemp('data').join('file.csv'))
    df.to_csv(filename)
    return filename

@pytest.fixture()
def test_df():
    df = pd.DataFrame({'col1': [1,2,3,4], 
                       'col2': [5,6,7,8],
                       'col3': [9,10,11,12]})
    return df

class Test_get_batch_generator:
    @pytest.mark.parametrize(
        'input_sequance, batch_size, output_sequance',
        [
            # input sequance bigger than batch size and can't be integer divided
            ([0,1,2,3,4,5], 4, [np.array([0,1,2,3]), np.array([4,5])]),
            # input sequance less than batch size and can't be integer divided
            ([0,1,2], 4, [np.array([0,1,2])]),
            # input sequance bigger than batch size and can be integer divided
            ([0,1,2,3], 2, [np.array([0,1]), np.array([2,3])]),
        ]
    )
    def test_correct_input_where_items_are_simple_data_type(self, input_sequance, batch_size, output_sequance):
        input_gen = (i for i in input_sequance)
        res_sequance = [batch for batch in get_batch_generator(input_gen, batch_size)]
        for res_seq, inp_seq in zip(res_sequance, output_sequance): 
            assert (res_seq == inp_seq).all()


    
    @pytest.mark.parametrize(
        'input_sequance, batch_size, output_sequance',
        [
            # input sequance bigger than batch size and can't be integer divided
            ([np.array([0,1]), np.array([2,3]), np.array([4,5])], 2, [np.array([[0,1], [2,3]]), np.array([4,5])]),
            # input sequance less than batch size and can't be integer divided
            ([np.array([0,1]), np.array([2,3]), np.array([4,5])], 4, [np.array([[0,1], [2,3], [4,5]])]),
            # input sequance bigger than batch size and can be integer divided
            ([np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7])], 2, [np.array([[0,1], [2,3]]), np.array([[4,5], [6,7]])]),
        ]
    )
    def test_correct_input_where_items_are_numpy_ndarrays(self, input_sequance, batch_size, output_sequance):
        input_gen = (i for i in input_sequance)
        res_sequance = [batch for batch in get_batch_generator(input_gen, batch_size)]
        for res_seq, inp_seq in zip(res_sequance, output_sequance): 
            assert (res_seq == inp_seq).all()

    @pytest.mark.parametrize(
        'input_sequance, batch_size',
        [
            (1, 1),
        ]
    )
    def test_wrong_type_input(self, input_sequance, batch_size):
        with pytest.raises(TypeError):
            assert get_batch_generator((i for i in input_sequance), batch_size)
    
            
class Test_get_x_and_y_data_dfs:
    pass

class Test_get_crop_generator:
    pass

class Test_get_augmented_crop_generator:
    pass

class Test__get_df_from_defects_file:
    pass
    
class Test__split_cell_string_value_to_numpy_array_of_64_values:

    def test_32_time_amp_pars_str_input_without_spaces(self):
        input_value = """
        25.8:37.947,26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6:-46.648,27.1:39.598,29.1:47.666,
        29.3:-54.845,29.4:52.46,29.5:-38.781,29.7:31.496,
        30.4:27.129,30.5:-35.327,30.7:34.409,32.2:42.332,
        32.3:-48.99,32.5:44.181,32.6:-30.463,33.5:-33.466,
        33.7:32,34.8:30.984,34.9:-29.394,35.2:36.222,
        35.3:-41.183,35.5:34.871,36.1:-27.713,38.3:29.933,
        38.4:-34.871,39.1:-26.533,41.5:-28.284,41.5:-28.284"""
        res = np.array([25.8, 26.1, 26.2, 26.3,
                        26.5, 26.6, 27.1, 29.1,
                        29.3, 29.4, 29.5, 29.7, 
                        30.4, 30.5, 30.7, 32.2,
                        32.3, 32.5, 32.6, 33.5,
                        33.7, 34.8, 34.9, 35.2,
                        35.3, 35.5, 36.1, 38.3,
                        38.4, 39.1, 41.5, 41.5,
                        37.947, -54.259, 61.709, -59.867,
                        55.136, -46.648, 39.598, 47.666,
                        -54.845, 52.46, -38.781, 31.496,
                        27.129, -35.327, 34.409, 42.332,
                        -48.99, 44.181, -30.463, -33.466,
                        32, 30.984, -29.394, 36.222,
                        -41.183, 34.871, -27.713, 29.933,
                        -34.871, -26.533, -28.284, -28.284]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input_value) == res).all()
        
    def test_32_time_amp_pars_str_input_with_spaces(self):
        input_value = """
        25.8:37.947, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6: -46.648,27.1:39.598,29.1:47.666,
        29.3:-54.845,29.4 :52.46,29.5:-38.781,29.7:31.496,
        30.4:27.129,30.5:-35.327 , 30.7:34.409,32.2:42.332,
        32.3 : -48.99,32.5:44.181,32.6:-30.463,33.5:-33.466,
        33.7:32,34.8:30.984,34.9:-29.394 , 35.2:36.222,
        35.3:-41.183,35.5:34.871,36.1:-27.713,38.3:29.933,
        38.4:-34.871,39.1:-26.533,41.5:-28.284,41.5:-28.284"""
        res = np.array([25.8, 26.1, 26.2, 26.3,
                        26.5, 26.6, 27.1, 29.1,
                        29.3, 29.4, 29.5, 29.7, 
                        30.4, 30.5, 30.7, 32.2,
                        32.3, 32.5, 32.6, 33.5,
                        33.7, 34.8, 34.9, 35.2,
                        35.3, 35.5, 36.1, 38.3,
                        38.4, 39.1, 41.5, 41.5,
                        37.947, -54.259, 61.709, -59.867,
                        55.136, -46.648, 39.598, 47.666,
                        -54.845, 52.46, -38.781, 31.496,
                        27.129, -35.327, 34.409, 42.332,
                        -48.99, 44.181, -30.463, -33.466,
                        32, 30.984, -29.394, 36.222,
                        -41.183, 34.871, -27.713, 29.933,
                        -34.871, -26.533, -28.284, -28.284]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input_value) == res).all()
        
    def test_31_time_amp_pars_str_input(self):
        input_value = """
        25.8:37.947, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6: -46.648,27.1:39.598,29.1:47.666,
        29.3:-54.845,29.4 :52.46,29.5:-38.781,29.7:31.496,
        30.4:27.129,30.5:-35.327 , 30.7:34.409,32.2:42.332,
        32.3 : -48.99,32.5:44.181,32.6:-30.463,33.5:-33.466,
        33.7:32,34.8:30.984,34.9:-29.394 , 35.2:36.222,
        35.3:-41.183,35.5:34.871,36.1:-27.713,38.3:29.933,
        38.4:-34.871,39.1:-26.533,41.5:-28.284"""
        res = np.array([0, 25.8, 26.1, 26.2, 
                        26.3, 26.5, 26.6, 27.1, 
                        29.1, 29.3, 29.4, 29.5, 
                        29.7, 30.4, 30.5, 30.7, 
                        32.2, 32.3, 32.5, 32.6, 
                        33.5, 33.7, 34.8, 34.9, 
                        35.2, 35.3, 35.5, 36.1, 
                        38.3, 38.4, 39.1, 41.5,
                        0, 37.947, -54.259, 61.709, 
                        -59.867, 55.136, -46.648, 39.598, 
                        47.666, -54.845, 52.46, -38.781, 
                        31.496, 27.129, -35.327, 34.409, 
                        42.332, -48.99, 44.181, -30.463, 
                        -33.466, 32, 30.984, -29.394, 
                        36.222, -41.183, 34.871, -27.713, 
                        29.933, -34.871, -26.533, -28.284]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input_value) == res).all()
    
    def test_34_time_amp_pars_str_input(self):
        input_value = """
        25.8:37.947, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6: -46.648,27.1:39.598,29.1:47.666,
        29.3:-54.845,29.4 :52.46,29.5:-38.781,29.7:31.496,
        30.4:27.129,30.5:-35.327 , 30.7:34.409,32.2:42.332,
        32.3 : -48.99,32.5:44.181,32.6:-30.463,33.5:-33.466,
        33.7:32,34.8:30.984,34.9:-29.394 , 35.2:36.222,
        35.3:-41.183,35.5:34.871,36.1:-27.713,38.3:29.933,
        38.4:-34.871,39.1:-26.533,41.5:-28.284,41.5:-28.284,
        41.5:-28.284,41.5:-28.284"""
        with pytest.raises(ValueError):
            assert _split_cell_string_value_to_numpy_array_of_64_values(input_value)
        
    def test_6_time_amp_pars_str_input(self):
        input_value = """
        25.8:37.947, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6: -46.648,"""
        res = np.array([0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 25.8, 26.1, 
                        26.2, 26.3, 26.5, 26.6,
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 37.947, -54.259, 
                        61.709, -59.867, 55.136, -46.648]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input_value) == res).all()
        
    def test_0_time_amp_pars_str_input(self):
        input_value = """--"""
        res = np.zeros((64)).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input_value) == res).all()
        
    def test_wrong_str_input(self):
        input_value = """fgreg tw345.345,4234;43tg5,l4y,45t,34t3q4ft4644;;546;:454--"""
        res = np.zeros((64)).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input_value) == res).all()
        
    def test_6_time_amp_pars_with_uncomplete_pars_str_input(self):
        input_value = """
        25.8:, 26.1:-54.259, 26.2:61.709, 26.3:-59.867,
        26.5:55.136, : -46.648,"""
        res = np.array([0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        26.1, 26.2, 26.3, 26.5,
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        -54.259, 61.709, -59.867, 55.136]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input_value) == res).all()
    

class Test__get_df_from_data_file:
    pass

if __name__ == "__main__":
    pytest.main()