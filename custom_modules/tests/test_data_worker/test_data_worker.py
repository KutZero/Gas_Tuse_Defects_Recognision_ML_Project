from custom_modules.data_worker.data_worker import get_x_and_y_data, \
    _df_to_image_like_numpy, calc_model_prediction_accuracy, \
    reshape_x_df_to_image_like_numpy, reshape_y_df_to_image_like_numpy, \
    normalize_data, standardize_data, split_def_and_non_def_data, \
    create_binary_arr_from_mask_arr, create_depth_arr_from_mask_arr, augment_data, \
    split_data_to_train_val_datasets, _calculate_crops_with_defects_positions, \
    _get_df_from_defects_file, _split_cell_string_value_to_numpy_array_of_64_values, \
    _get_df_from_data_file

import pytest
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from contextlib import nullcontext as does_not_raise
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field

class Test_get_x_and_y_data:
    pass

class Test_calc_model_prediction_accuracy:
    pass

class Test__df_to_image_like_numpy:
    def test_correct_input(self):
        input_df = pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                           'col2': [np.array([5.,6.]), np.array([7.,8.])]})
        res = np.array([[[1.,2.],[5.,6.]],
                        [[3.,4.],[7.,8.]]])
        assert (_df_to_image_like_numpy(input_df) == res).all()

    @pytest.mark.parametrize(
        'input_df, expectation',
        [
            (pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                           'col2': ['test_string', np.array([7.,8.])]}), pytest.raises(TypeError)),
            (pd.DataFrame({'col1': [np.array([1.,2.]), 32], 
                           'col2': [np.array([5.,6.]), np.array([7.,8.])]}), pytest.raises(TypeError)),
            (pd.DataFrame({'col1': [[1,2,3,4], np.array([3.,4.])], 
                           'col2': [np.array([5.,6.]), np.array([7.,8.])]}), pytest.raises(TypeError)),
            (pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                           'col2': [np.array([5.,6.]), None]}), pytest.raises(TypeError)),
        ]
    )
    def test_uncorrect_input_cell_value_is_not_numpy_array(self, input_df, expectation):
        with expectation:
            assert _df_to_image_like_numpy(input_df)

    def test_uncorrect_input_cell_value_is_not_flat_numpy_array(self):
        input_df = pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                              'col2': [np.array([[5.,6.]]), np.array([7.,8.])]})
        with pytest.raises(ValueError):
            _df_to_image_like_numpy(input_df)
    
    def test_uncorrect_input_cell_value_is_numpy_array_with_not_float_values(self):
        input_df = pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                                 'col2': [np.array([5,6]), np.array([7.,8.])]})
        with pytest.raises(TypeError):
            _df_to_image_like_numpy(input_df)

@pytest.fixture()
def test_df():
    df = pd.DataFrame({'col1': [1,2,3,4], 
                       'col2': [5,6,7,8],
                       'col3': [9,10,11,12]})
    return df

class Test_reshape_x_df_to_image_like_numpy:
    pass

class Test_reshape_y_df_to_image_like_numpy:
    pass

class Test_normalize_data:
    pass

class Test_standardize_data:
    pass

class Test_split_def_and_non_def_data:
    pass

class Test_create_binary_arr_from_mask_arr:
    def test_correct_input(self):
        # 3 masks of size 2*2 and 1 color channel
        input = np.array([[
                    [[1.],[2.]],[[3.],[4.]]],
                    [[[0],[0.]],[[0],[0]]],
                    [[[9.],[10.]],[[11.],[12.]]]])
        
        res = np.array([True, False, True])
        assert (create_binary_arr_from_mask_arr(input) == res).all()
    
    def test_uncorrect_input_value_type(self):
        with pytest.raises(ValidationError):
            create_binary_arr_from_mask_arr('string')
            
    def test_uncorrect_input_array_shape(self):
        input = np.array([[1,2],[3,4]])
        with pytest.raises(ValueError):
            create_binary_arr_from_mask_arr(input)

    def test_uncorrect_input_array_dtype(self):
        input = np.array([[
            [['1'],['2.']],[['3.'],['4.']]]])
        with pytest.raises(ValueError):
            create_binary_arr_from_mask_arr(input)

class Test_create_depth_arr_from_mask_arr:
    def test_correct_input(self):
        # 3 masks of size 2*2 and 1 color channel
        input = np.array([[
                    [[1.],[2.]],[[3.],[4.]]],
                    [[[0],[0.]],[[0],[0]]],
                    [[[9.],[10.]],[[11.],[12.]]]])
        
        res = np.array([4., 0., 12.])
        assert (create_depth_arr_from_mask_arr(input) == res).all()
    
    def test_uncorrect_input_value_type(self):
        with pytest.raises(ValidationError):
            create_depth_arr_from_mask_arr('string')
            
    def test_uncorrect_input_array_shape(self):
        input = np.array([[1,2],[3,4]])
        with pytest.raises(ValueError):
            create_depth_arr_from_mask_arr(input)

    def test_uncorrect_input_array_dtype(self):
        input = np.array([[
            [['1'],['2.']],[['3.'],['4.']]]])
        with pytest.raises(ValueError):
            create_depth_arr_from_mask_arr(input)

class Test_augment_data:
    pass
    
class Test_split_data_to_train_val_datasets:
    pass
    
class Test__calculate_crops_with_defects_positions:
    pass

class Test__get_df_from_defects_file:
    pass
    
class Test__split_cell_string_value_to_numpy_array_of_64_values:

    def test_32_time_amp_pars_str_input_without_spaces(self):
        input = """
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
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
        
    def test_32_time_amp_pars_str_input_with_spaces(self):
        input = """
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
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
        
    def test_31_time_amp_pars_str_input(self):
        input = """
        25.8:37.947, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6: -46.648,27.1:39.598,29.1:47.666,
        29.3:-54.845,29.4 :52.46,29.5:-38.781,29.7:31.496,
        30.4:27.129,30.5:-35.327 , 30.7:34.409,32.2:42.332,
        32.3 : -48.99,32.5:44.181,32.6:-30.463,33.5:-33.466,
        33.7:32,34.8:30.984,34.9:-29.394 , 35.2:36.222,
        35.3:-41.183,35.5:34.871,36.1:-27.713,38.3:29.933,
        38.4:-34.871,39.1:-26.533,41.5:-28.284"""
        res = np.array([25.8, 26.1, 26.2, 26.3,
                        26.5, 26.6, 27.1, 29.1,
                        29.3, 29.4, 29.5, 29.7, 
                        30.4, 30.5, 30.7, 32.2,
                        32.3, 32.5, 32.6, 33.5,
                        33.7, 34.8, 34.9, 35.2,
                        35.3, 35.5, 36.1, 38.3,
                        38.4, 39.1, 41.5, 0,
                        37.947, -54.259, 61.709, -59.867,
                        55.136, -46.648, 39.598, 47.666,
                        -54.845, 52.46, -38.781, 31.496,
                        27.129, -35.327, 34.409, 42.332,
                        -48.99, 44.181, -30.463, -33.466,
                        32, 30.984, -29.394, 36.222,
                        -41.183, 34.871, -27.713, 29.933,
                        -34.871, -26.533, -28.284, 0]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
    
    def test_34_time_amp_pars_str_input(self):
        input = """
        25.8:37.947, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6: -46.648,27.1:39.598,29.1:47.666,
        29.3:-54.845,29.4 :52.46,29.5:-38.781,29.7:31.496,
        30.4:27.129,30.5:-35.327 , 30.7:34.409,32.2:42.332,
        32.3 : -48.99,32.5:44.181,32.6:-30.463,33.5:-33.466,
        33.7:32,34.8:30.984,34.9:-29.394 , 35.2:36.222,
        35.3:-41.183,35.5:34.871,36.1:-27.713,38.3:29.933,
        38.4:-34.871,39.1:-26.533,41.5:-28.284,41.5:-28.284,
        41.5:-28.284,41.5:-28.284"""
        res = np.array([25.8, 26.1, 26.2, 26.3,
                        26.5, 26.6, 27.1, 29.1,
                        29.3, 29.4, 29.5, 29.7, 
                        30.4, 30.5, 30.7, 32.2,
                        32.3, 32.5, 32.6, 33.5,
                        33.7, 34.8, 34.9, 35.2,
                        35.3, 35.5, 36.1, 38.3,
                        38.4, 39.1, 41.5, 41.5,
                        41.5, 41.5,
                        37.947, -54.259, 61.709, -59.867,
                        55.136, -46.648, 39.598, 47.666,
                        -54.845, 52.46, -38.781, 31.496,
                        27.129, -35.327, 34.409, 42.332,
                        -48.99, 44.181, -30.463, -33.466,
                        32, 30.984, -29.394, 36.222,
                        -41.183, 34.871, -27.713, 29.933,
                        -34.871, -26.533, -28.284, -28.284,
                        -28.284, -28.284]).astype(float)
        with pytest.raises(ValueError):
            assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
        
    def test_6_time_amp_pars_str_input(self):
        input = """
        25.8:37.947, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,26.6: -46.648,"""
        res = np.array([25.8, 26.1, 26.2, 26.3,
                        26.5, 26.6, 0, 0,
                        0, 0, 0, 0, 
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        37.947, -54.259, 61.709, -59.867,
                        55.136, -46.648, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
        
    def test_0_time_amp_pars_str_input(self):
        input = """--"""
        res = np.zeros((64)).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
        
    def test_wrong_str_input(self):
        input = """fgreg tw345.345,4234;43tg5,l4y,45t,34t3q4ft4644;;546;:454--"""
        res = np.zeros((64)).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
        
    def test_6_time_amp_pars_with_uncomplete_pars_str_input(self):
        input = """
        25.8:, 26.1:-54.259,26.2:61.709,26.3:-59.867,
        26.5:55.136,: -46.648,"""
        res = np.array([0, 26.1, 26.2, 26.3,
                        26.5, 0, 0, 0,
                        0, 0, 0, 0, 
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, -54.259, 61.709, -59.867,
                        55.136, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,]).astype(float)
        assert (_split_cell_string_value_to_numpy_array_of_64_values(input) == res).all()
    

class Test__get_df_from_data_file:
    pass

@pytest.fixture
def csv_file_path(tmpdir_factory):
    df = pd.DataFrame({'col1': [1, 2, 1], 
                       'col2': [3, 4, 1],
                       'col3': [5, 6, 1]}) 
    filename = str(tmpdir_factory.mktemp('data').join('file.csv'))
    df.to_csv(filename)
    return filename

if __name__ == "__main__":
    pytest.main()