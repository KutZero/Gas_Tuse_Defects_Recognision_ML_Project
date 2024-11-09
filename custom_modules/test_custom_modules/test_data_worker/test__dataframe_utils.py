from custom_modules.data_worker._dataframe_utils import (
    roll_df, 
    df_to_numpy,
    match_df_for_crops_dividing, 
    extend_df_for_prediction,
    _check_df_cell_is_correct_numpy_array)

import pytest
import pandas as pd
import numpy as np
import re
import os

from contextlib import nullcontext as does_not_raise
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator

class Test_roll_df:
    pass

@pytest.fixture()
def test_input_df():
    df = pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3])
    return df

class Test__check_df_cell_is_correct_numpy_array:
    pass

class Test_df_to_numpy:
    def test_correct_input(self):
        input_df = pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                           'col2': [np.array([5.,6.]), np.array([7.,8.])]})
        res = np.array([[[1.,2.],[5.,6.]],
                        [[3.,4.],[7.,8.]]])
        assert (df_to_numpy(input_df) == res).all()

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
            assert df_to_numpy(input_df)

    def test_uncorrect_input_cell_value_is_not_flat_numpy_array(self):
        input_df = pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                              'col2': [np.array([[5.,6.]]), np.array([7.,8.])]})
        with pytest.raises(ValueError):
            df_to_numpy(input_df)
    
    def test_uncorrect_input_cell_value_is_numpy_array_with_not_float_values(self):
        input_df = pd.DataFrame({'col1': [np.array([1.,2.]), np.array([3.,4.])], 
                                 'col2': [np.array([5,6]), np.array([7.,8.])]})
        with pytest.raises(TypeError):
            df_to_numpy(input_df)

class Test_match_df_for_crops_dividing:
    @pytest.mark.parametrize(
    'crop_size, crop_step, mode, res',
    [
        # nothing
        (3, 1, 'extend', pd.DataFrame(data=[[1,2,3],
                                            [4,5,6],
                                            [7,8,9],
                                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3])),
        # nothing
        ('3', 1, 'extend', pd.DataFrame(data=[[1,2,3],
                                              [4,5,6],
                                              [7,8,9],
                                              [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3])),
        # nothing
        (3, '1', 'extend', pd.DataFrame(data=[[1,2,3],
                                              [4,5,6],
                                              [7,8,9],
                                              [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3])),
        # nothing
        ('3', '1', 'extend', pd.DataFrame(data=[[1,2,3],
                                                [4,5,6],
                                                [7,8,9],
                                                [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3])),
        # add rows only
        (3, 2, 'extend', pd.DataFrame(data=[[1,2,3],
                                            [4,5,6],
                                            [7,8,9],
                                            [10,11,12],
                                            [7,8,9]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3,2])),
        # add cols only
        (2, 2, 'extend', pd.DataFrame(data=[[1,2,3,2],
                                            [4,5,6,5],
                                            [7,8,9,8],
                                            [10,11,12,11]], 
                      columns=['col1','col2','col3','col2'],
                      index=[0,1,2,3])),
        # add rows and cols
        (2, 3, 'extend', pd.DataFrame(data=[[1,2,3,2,1],
                                            [4,5,6,5,4],
                                            [7,8,9,8,7],
                                            [10,11,12,11,10],
                                            [7,8,9,8,7]], 
                      columns=['col1','col2','col3','col2','col1'],
                      index=[0,1,2,3,2])),
         # remove rows only
        (3, 2, 'crop', pd.DataFrame(data=[[1,2,3],
                                          [4,5,6],
                                          [7,8,9]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2])),
        # remove cols only
        (2, 2, 'crop', pd.DataFrame(data=[[1,2],
                                          [4,5],
                                          [7,8],
                                          [10,11]], 
                      columns=['col1','col2'],
                      index=[0,1,2,3])),
        # removev rows and cols
        (2, 3, 'crop', pd.DataFrame(data=[[1,2],
                                          [4,5]], 
                      columns=['col1','col2'],
                      index=[0,1])),
    ]
    )
    def test_correct_input_extend_principle(self, test_input_df, crop_size, crop_step, mode, res):
        assert match_df_for_crops_dividing(test_input_df, crop_size=crop_size, crop_step=crop_step, mode=mode).equals(res) == True

    @pytest.mark.parametrize(
    'crop_size, crop_step, mode, expectation',
    [
        # crop_size bigger than rows quantity 
        (5, 1, 'extend', pytest.raises(ValueError)),
        # crop_size bigger than cols quantity
        (4, 1, 'extend', pytest.raises(ValueError)),
        # crop_step bigger than rows quantity
        (2, 5, 'extend', pytest.raises(ValueError)),
        # crop_step bigger than cols quantity
        (2, 4, 'extend', pytest.raises(ValueError)),
        # negative crop_size
        (-2, 5, 'extend', pytest.raises(ValueError)),
        # negative crop_step
        (2, -2, 'extend', pytest.raises(ValueError)),
        # zero crop_size
        (0, 2, 'extend', pytest.raises(ValueError)),
        # zero crop_step
        (2, 0, 'extend', pytest.raises(ValueError)),
        # mode is uncorrect word
        (2, 0, 'sdfasdf', pytest.raises(ValueError))
    ]
    )
    def test_uncorrect_crop_size_and_crop_step(self, test_input_df, crop_size, crop_step, mode, expectation):
        with expectation:
            assert match_df_for_crops_dividing(test_input_df, crop_size=crop_size, crop_step=crop_step, mode=mode)
            
    @pytest.mark.parametrize(
    'input_df, crop_size, crop_step, mode, expectation',
    [
        # df is not pandas.DataFrame
        ('sdfsdf', 5, 1, 'extend', pytest.raises(ValidationError)),
        # crop_size is not int
        (pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3]), '2grg', 1, 'extend', pytest.raises(ValidationError)),
        # crop_step in not int
        (pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3]), 2, 'reger', 'extend', pytest.raises(ValidationError)),
        # mode in not str
        (pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3]), 2, 2, 5, pytest.raises(ValidationError))
    ]
    )
    def test_uncorrect_input_values_type(self, input_df, crop_size, crop_step, mode, expectation):
        with expectation:
            assert match_df_for_crops_dividing(input_df, crop_size=crop_size, crop_step=crop_step, mode=mode)

class Test_extend_df_for_prediction:
    
    @pytest.mark.parametrize(
    'crop_size, only_horizontal, res',
    [
        (2, False, pd.DataFrame(data=[[6,4,5,6,4],
                               [3,1,2,3,1],
                               [6,4,5,6,4],
                               [9,7,8,9,7],
                               [12,10,11,12,10],
                               [9,7,8,9,7]], 
                      columns=['col3','col1','col2','col3','col1'],
                      index=[1,0,1,2,3,2])),

        (3, False, pd.DataFrame(data=[[8,9,7,8,9,7,8],
                               [5,6,4,5,6,4,5],
                               [2,3,1,2,3,1,2],
                               [5,6,4,5,6,4,5],
                               [8,9,7,8,9,7,8],
                               [11,12,10,11,12,10,11],
                               [8,9,7,8,9,7,8],
                               [5,6,4,5,6,4,5]], 
                      columns=['col2','col3','col1','col2','col3','col1','col2'],
                      index=[2,1,0,1,2,3,2,1])),
        
        (2, True, pd.DataFrame(data=[[3,1,2,3,1],
                               [6,4,5,6,4],
                               [9,7,8,9,7],
                               [12,10,11,12,10]], 
                      columns=['col3','col1','col2','col3','col1'],
                      index=[0,1,2,3])),
    ]
    )
    def test_correct_input_extend_principle(self, test_input_df, crop_size, only_horizontal, res):
        assert extend_df_for_prediction(test_input_df, crop_size=crop_size, only_horizontal=only_horizontal).equals(res) == True
        
    @pytest.mark.parametrize(
    'crop_size, expectation',
    [
        # crop_size bigger than rows quantity 
        (5, pytest.raises(ValueError)),
        # crop_size bigger than cols quantity
        (4, pytest.raises(ValueError)),
        # negative crop_size
        (-2, pytest.raises(ValueError)),
        # zero crop_size
        (0, pytest.raises(ValueError))
    ]
    )
    def test_uncorrect_crop_size_and_crop_step(self, test_input_df, crop_size, expectation):
        with expectation:
            assert extend_df_for_prediction(test_input_df, crop_size=crop_size)
            
    @pytest.mark.parametrize(
    'input_df, crop_size, expectation',
    [
        # df is not pandas.DataFrame
        ('sdfsdf', 2, pytest.raises(ValidationError)),
        # crop_size is not int
        (pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3]), 'gfg2', pytest.raises(ValidationError))
    ]
    )
    def test_uncorrect_input_values_type(self, input_df, crop_size, expectation):
        with expectation:
            assert extend_df_for_prediction(input_df, crop_size=crop_size)


if __name__ == "__main__":
    pytest.main()