from custom_modules.dataset import (
    get_crop_generator,
    get_x_and_y_data_dfs,
    _read_data_df_and_defects_df,
    _unify_dfs,
    _crop_data_df_and_defects_df,
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
    df = pd.DataFrame({'col1': [1,2,1], 
                       'col2': [3,4,1],
                       'col3': [5,6,1]}) 
    filename = str(tmpdir_factory.mktemp('data').join('file.csv'))
    df.to_csv(filename)
    return filename


@pytest.fixture()
def test_df():
    df = pd.DataFrame({'col1': [1,2,3,4], 
                       'col2': [5,6,7,8],
                       'col3': [9,10,11,12]})
    return df
    
            
class Test_get_x_and_y_data_dfs:
    pass


class Test__read_data_df_and_defects_df:
    pass


class Test__unify_dfs:
    pass


class Test__crop_data_df_and_defects_df:
    pass


class Test__get_df_from_defects_file:
    pass


class Test__get_df_from_data_file:
    pass


class Test_get_crop_generator:
    @pytest.mark.parametrize(
        'input_arr, crop_size, crop_step, augmentations, valid_res',
        [

            # input_arr can't be divided by crop_size with crop_step, redundant rows and cols ignored
            (np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]]), 2, 2, False, [np.array([[1,2],[4,5]]),
                                                     np.array([[7,8],[10,11]])]),

            # input_arr can be divided by crop_size with crop_step
            (np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]]), 2, 1, False, [np.array([[1,2], [4,5]]),
                                                     np.array([[2,3], [5,6]]),
                                                     np.array([[4,5], [7,8]]),
                                                     np.array([[5,6], [8,9]]),
                                                     np.array([[7,8], [10,11]]),
                                                     np.array([[8,9], [11,12]])]),
            
            #input_arr can be divided by crop_size with crop_step. Augmentations used
            (np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]), 2, 1, True, [np.array([[1,2], [4,5]]),
                                                 np.array([[2,3], [5,6]]),
                                                 np.array([[4,5], [7,8]]),
                                                 np.array([[5,6], [8,9]]),
                                                 np.array([[7,8], [4,5]]),
                                                 np.array([[8,9], [5,6]]),
                                                 np.array([[4,5], [1,2]]),
                                                 np.array([[5,6], [2,3]]),
                                                 np.array([[3,2], [6,5]]),
                                                 np.array([[2,1], [5,4]]),
                                                 np.array([[6,5], [9,8]]),
                                                 np.array([[5,4], [8,7]]),
                                                 np.array([[9,8], [6,5]]),
                                                 np.array([[8,7], [5,4]]),
                                                 np.array([[6,5], [3,2]]),
                                                 np.array([[5,4], [2,1]]),
                                                 np.array([[3,6], [2,5]]),
                                                 np.array([[6,9], [5,8]]),
                                                 np.array([[2,5], [1,4]]),
                                                 np.array([[5,8], [4,7]]),
                                                 np.array([[9,6], [8,5]]),
                                                 np.array([[6,3], [5,2]]),
                                                 np.array([[8,5], [7,4]]),
                                                 np.array([[5,2], [4,1]]),
                                                 np.array([[1,4], [2,5]]),
                                                 np.array([[4,7], [5,8]]),
                                                 np.array([[2,5], [3,6]]),
                                                 np.array([[5,8], [6,9]]),
                                                 np.array([[7,4], [8,5]]),
                                                 np.array([[4,1], [5,2]]),
                                                 np.array([[8,5], [9,6]]),
                                                 np.array([[5,2], [6,3]])]),
            
            #input_arr can't be divided by crop_size with crop_step. Augmentations used
            (np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]), 2, 2, True, [np.array([[1,2], [4,5]]),
                                                 np.array([[4,5], [1,2]]),
                                                 np.array([[2,1], [5,4]]),
                                                 np.array([[5,4], [2,1]]),
                                                 np.array([[2,5], [1,4]]),
                                                 np.array([[5,2], [4,1]]),
                                                 np.array([[1,4], [2,5]]),
                                                 np.array([[4,1], [5,2]])]),
            # input_arr can't be divided by crop_size with crop_step, redundant rows and cols ignored.
            # The input_arr has first 2 non empty dims and 1 empty one
            (np.array([[[1], [2], [3]],
                       [[4], [5], [6]]]), 2, 1, False, [np.array([[[1],[2]], [[4],[5]]]),
                                                        np.array([[[2],[3]], [[5],[6]]])]),

    ]
    )
    def test_correct_input(self, input_arr, crop_size, crop_step, augmentations, valid_res):
        get_res = list(get_crop_generator(input_arr, crop_size, crop_step, augmentations))
        
        assert len(get_res) == len(valid_res)
        
        for valid_res_item, get_res_item in zip(valid_res, get_res):
            assert (valid_res_item == get_res_item).all()

   
    @pytest.mark.parametrize(
    'input_arr, crop_size, crop_step',
    [
        # crop_size bigger than rows quantity 
        (np.array([[1, 2, 3, 4],
                   [4, 5, 6, 6],
                   [7, 8, 9, 6]]), 4, 1),
        # crop_size bigger than cols quantity
        (np.array([[1, 2],
                   [4, 5],
                   [7, 8]]), 3, 1),
        # crop_step bigger than rows quantity
        (np.array([[1, 2, 3, 4],
                   [4, 5, 6, 4],
                   [7, 8, 9, 4]]), 1, 4),
        # crop_step bigger than cols quantity
        (np.array([[1, 2],
                   [4, 5],
                   [7, 8]]), 1, 3),
        # crop_size bigger than rows and cols quantity
        (np.array([[1, 2,3],
                   [4, 5, 6],
                   [7, 8, 9]]), 4, 1),
        # crop_step bigger than rows and cols quantity
        (np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]), 1, 4),
        
        # input_arr has less than 2 dims
        (np.array([1, 2, 3]), 1, 1),
        # input_arr has less than 2 non empty dims
        (np.array([[1, 2, 3]]), 1, 1),
        # input_arr first dim is empty 
        (np.array([[[1 ,2, 3],
                    [1 ,2, 3]]]), 1, 1),
        # input_arr second dim is empty 
        (np.array([[[1, 2, 3]],
                   [[1, 2, 3]]]), 1, 1),
    ]
    )
    def test_uncorrect_value_input(self, input_arr, crop_size, crop_step):
        with pytest.raises(ValueError):
            next(get_crop_generator(input_arr, crop_size, crop_step))

    @pytest.mark.parametrize(
    'input_arr, crop_size, crop_step, augmentations',
    [
        # input_arr is not np.array
        ('test', 4, 1, False),
        # crop_size is not int
        (np.array([[1, 2],
                   [4, 5],
                   [7, 8]]), 'test', 1, False),
        # crop_step is not int
        (np.array([[1, 2, 3, 4],
                   [4, 5, 6, 4],
                   [7, 8, 9, 4]]), 1, 'test', False),
        # augmentations  is not bool
        (np.array([[1, 2],
                   [4, 5],
                   [7, 8]]),  1, 3, 'test'),
    ]
    )
    def test_uncorrect_type_input(self, input_arr, crop_size, crop_step, augmentations):
        with pytest.raises(ValidationError):
            next(get_crop_generator(input_arr, crop_size, crop_step, augmentations))


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


if __name__ == "__main__":
    pytest.main()