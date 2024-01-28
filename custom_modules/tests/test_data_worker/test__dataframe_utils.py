from custom_modules._dataframe_utils import roll_df, \
    extend_df_for_crops_dividing, extend_df_for_prediction

import pytest
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from contextlib import nullcontext as does_not_raise


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

class Test_extend_df_for_crops_dividing:
    @pytest.mark.parametrize(
    'crop_size, crop_step, res',
    [
        # nothing
        (3, 1, pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3])),

        # add rows only
        (3, 2, pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12],
                            [7,8,9]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3,2])),

        # add cols only
        (2, 2, pd.DataFrame(data=[[1,2,3,2],
                            [4,5,6,5],
                            [7,8,9,8],
                            [10,11,12,11]], 
                      columns=['col1','col2','col3','col2'],
                      index=[0,1,2,3])),

        # add rows and cols
        (2, 3, pd.DataFrame(data=[[1,2,3,2,1],
                            [4,5,6,5,4],
                            [7,8,9,8,7],
                            [10,11,12,11,10],
                            [7,8,9,8,7]], 
                      columns=['col1','col2','col3','col2','col1'],
                      index=[0,1,2,3,2])),
    ]
    )
    def test_correct_input_extend_principle(self, test_input_df, crop_size, crop_step, res):
        assert extend_df_for_crops_dividing(test_input_df, crop_size=crop_size, crop_step=crop_step).equals(res) == True

    @pytest.mark.parametrize(
    'crop_size, crop_step, expectation',
    [
        # crop_size bigger than rows quantity 
        (5, 1, pytest.raises(ValueError)),
        # crop_size bigger than cols quantity
        (4, 1, pytest.raises(ValueError)),
        # crop_step bigger than rows quantity
        (2, 5, pytest.raises(ValueError)),
        # crop_step bigger than cols quantity
        (2, 4, pytest.raises(ValueError)),
        # negative crop_size
        (-2, 5, pytest.raises(ValueError)),
        # negative crop_step
        (2, -2, pytest.raises(ValueError)),
        # zero crop_size
        (0, 2, pytest.raises(ValueError)),
        # zero crop_step
        (2, 0, pytest.raises(ValueError))
    ]
    )
    def test_uncorrect_crop_size_and_crop_step(self, test_input_df, crop_size, crop_step, expectation):
        with expectation:
            assert extend_df_for_crops_dividing(test_input_df, crop_size=crop_size, crop_step=crop_step)
            
    @pytest.mark.parametrize(
    'input_df, crop_size, crop_step, expectation',
    [
        # df is not pandas.DataFrame
        ('sdfsdf', 5, 1, pytest.raises(TypeError)),
        # crop_size is not int
        (pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3]), '2', 1, pytest.raises(TypeError)),
        # crop_step in not int
        (pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3]), 2, 'reger', pytest.raises(TypeError))
    ]
    )
    def test_uncorrect_input_values_type(self, input_df, crop_size, crop_step, expectation):
        with expectation:
            assert extend_df_for_crops_dividing(input_df, crop_size=crop_size, crop_step=crop_step)

class Test_extend_df_for_prediction:
    
    @pytest.mark.parametrize(
    'crop_size, res',
    [
        (2, pd.DataFrame(data=[[6,4,5,6,4],
                               [3,1,2,3,1],
                               [6,4,5,6,4],
                               [9,7,8,9,7],
                               [12,10,11,12,10],
                               [9,7,8,9,7]], 
                      columns=['col3','col1','col2','col3','col1'],
                      index=[1,0,1,2,3,2])),

        (3, pd.DataFrame(data=[[8,9,7,8,9,7,8],
                               [5,6,4,5,6,4,5],
                               [2,3,1,2,3,1,2],
                               [5,6,4,5,6,4,5],
                               [8,9,7,8,9,7,8],
                               [11,12,10,11,12,10,11],
                               [8,9,7,8,9,7,8],
                               [5,6,4,5,6,4,5]], 
                      columns=['col2','col3','col1','col2','col3','col1','col2'],
                      index=[2,1,0,1,2,3,2,1]))
    ]
    )
    def test_correct_input_extend_principle(self, test_input_df, crop_size, res):
        assert extend_df_for_prediction(test_input_df, crop_size=crop_size).equals(res) == True
        
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
        ('sdfsdf', 2, pytest.raises(TypeError)),
        # crop_size is not int
        (pd.DataFrame(data=[[1,2,3],
                            [4,5,6],
                            [7,8,9],
                            [10,11,12]], 
                      columns=['col1','col2','col3'],
                      index=[0,1,2,3]), '2', pytest.raises(TypeError))
    ]
    )
    def test_uncorrect_input_values_type(self, input_df, crop_size, expectation):
        with expectation:
            assert extend_df_for_prediction(input_df, crop_size=crop_size)


if __name__ == "__main__":
    pytest.main()