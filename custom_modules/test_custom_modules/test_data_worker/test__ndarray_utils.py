from custom_modules.data_worker._ndarray_utils import (
    match_ndarray_for_crops_dividing,
    extend_ndarray_for_prediction
)

import pytest
import pandas as pd
import numpy as np
import re
import os

from contextlib import nullcontext as does_not_raise
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator


@pytest.fixture()
def test_input_arr():
    arr = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9],
                   [10,11,12]])
    return arr


class Test_match_ndarray_for_crops_dividing:
    @pytest.mark.parametrize(
    'crop_size, crop_step, mode, res',
    [
        # nothing
        (3, 1, 'extend', np.array([[1,2,3],
                                   [4,5,6],
                                   [7,8,9],
                                   [10,11,12]])),
        # nothing
        ('3', 1, 'extend', np.array([[1,2,3],
                                     [4,5,6],
                                     [7,8,9],
                                     [10,11,12]])),
        # nothing
        (3, '1', 'extend', np.array([[1,2,3],
                                     [4,5,6],
                                     [7,8,9],
                                     [10,11,12]])),
        # nothing
        ('3', '1', 'extend', np.array([[1,2,3],
                                       [4,5,6],
                                       [7,8,9],
                                       [10,11,12]])),
        # nothing
        (3, 1, 'crop', np.array([[1,2,3],
                                 [4,5,6],
                                 [7,8,9],
                                 [10,11,12]])),
        # nothing
        ('3', 1, 'crop', np.array([[1,2,3],
                                   [4,5,6],
                                   [7,8,9],
                                   [10,11,12]])),
        # nothing
        (3, '1', 'crop', np.array([[1,2,3],
                                   [4,5,6],
                                   [7,8,9],
                                   [10,11,12]])),
        # nothing
        ('3', '1', 'crop', np.array([[1,2,3],
                                     [4,5,6],
                                     [7,8,9],
                                     [10,11,12]])),
        # add rows only
        (3, 2, 'extend', np.array([[1,2,3],
                                   [4,5,6],
                                   [7,8,9],
                                   [10,11,12],
                                   [7,8,9]])),
        # add cols only
        (2, 2, 'extend', np.array([[1,2,3,2],
                                   [4,5,6,5],
                                   [7,8,9,8],
                                   [10,11,12,11]])),
        # add rows and cols
        (2, 3, 'extend', np.array([[1,2,3,2,1],
                                   [4,5,6,5,4],
                                   [7,8,9,8,7],
                                   [10,11,12,11,10],
                                   [7,8,9,8,7]])),
        # remove rows only
        (3, 2, 'crop', np.array([[1,2,3],
                                 [4,5,6],
                                 [7,8,9]])),
        # remove cols only
        (2, 2, 'crop', np.array([[1,2],
                                 [4,5],
                                 [7,8],
                                 [10,11]])),
        # remove rows and cols
        (2, 3, 'crop', np.array([[1,2],
                                 [4,5]])),
    ]
    )
    def test_correct_input_extend_principle(self, test_input_arr, crop_size, crop_step, mode, res):
        output = match_ndarray_for_crops_dividing(test_input_arr, crop_size=crop_size, 
                                                  crop_step=crop_step, mode=mode)
        assert (output == res).all()

    
    @pytest.mark.parametrize(
    'input_arr, crop_size, crop_step',
    [
        (np.array([[[1],[2],[3]],
                  [[4],[5],[6]],
                  [[7],[8],[9]]]),1,1),
    ]
    )
    def test_correct_input_but_input_array_has_empty_dims(self, input_arr, crop_size, crop_step):
        assert (match_ndarray_for_crops_dividing(input_arr, crop_size, crop_step) == input_arr).all()

    
    @pytest.mark.parametrize(
    'input_arr, crop_size, crop_step',
    [
        # input_arr has less than 2 non empty dims
        (np.array([[1,2,3]]),1,1),
        # input_arr first dim is empty 
        (np.array([[[1,2,3],
                    [1,2,3]]]),1,1),
        # input_arr second dim is empty 
        (np.array([[[1,2,3]],
                   [[1,2,3]]]),1,1),
    ]
    )
    def test_uncorrect_input_when_input_array_has_empty_dims(self, input_arr, crop_size, crop_step):
        with pytest.raises(ValueError):
            match_ndarray_for_crops_dividing(input_arr, crop_size, crop_step)

    
    @pytest.mark.parametrize(
    'crop_size, crop_step, mode',
    [
        # crop_size bigger than rows quantity 
        (5, 1, 'extend'),
        # crop_size bigger than cols quantity
        (4, 1, 'extend'),
        # crop_step bigger than rows quantity
        (2, 5, 'extend'),
        # crop_step bigger than cols quantity
        (2, 4, 'extend'),
        # negative crop_size
        (-2, 5, 'extend'),
        # negative crop_step
        (2, -2, 'extend'),
        # zero crop_size
        (0, 2, 'extend'),
        # zero crop_step
        (2, 0, 'extend'),
        # uncorrect mode name
        (2, 0, 'esdfsdfsadf'),
    ]
    )
    def test_uncorrect_crop_size_and_crop_step(self, test_input_arr, crop_size, crop_step, mode):
        with pytest.raises(ValueError):
            match_ndarray_for_crops_dividing(test_input_arr, crop_size=crop_size, 
                                             crop_step=crop_step, mode=mode)

    
    @pytest.mark.parametrize(
    'input_arr, crop_size, crop_step, mode',
    [
        # array is not numpy.ndarray
        ('sdfsdf', 5, 1, 'extend'),
        # crop_size is not int
        (np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9],
                   [10,11,12]]), '2grg', 1, 'extend'),
        # crop_step in not int
        (np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9],
                   [10,11,12]]), 2, 'reger', 'extend'),
        # mode in not str
        (np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9],
                   [10,11,12]]), 2, 1, 3)
    ]
    )
    def test_uncorrect_input_values_type(self, input_arr, crop_size, crop_step, mode):
        with pytest.raises(ValidationError):
            assert (match_ndarray_for_crops_dividing(input_arr, crop_size=crop_size, 
                                                     crop_step=crop_step, mode=mode)).all()


class Test_extend_ndarray_for_prediction:
    @pytest.mark.parametrize(
    'crop_size, only_horizontal, res',
    [
        (2, False, np.array([[6,4,5,6,4],
                               [3,1,2,3,1],
                               [6,4,5,6,4],
                               [9,7,8,9,7],
                               [12,10,11,12,10],
                               [9,7,8,9,7]])),

        (3, False, np.array([[8,9,7,8,9,7,8],
                               [5,6,4,5,6,4,5],
                               [2,3,1,2,3,1,2],
                               [5,6,4,5,6,4,5],
                               [8,9,7,8,9,7,8],
                               [11,12,10,11,12,10,11],
                               [8,9,7,8,9,7,8],
                               [5,6,4,5,6,4,5]])),
        
        (2, True, np.array([[3,1,2,3,1],
                               [6,4,5,6,4],
                               [9,7,8,9,7],
                               [12,10,11,12,10]])),
    ]
    )
    def test_correct_input_extend_principle(self, test_input_arr, crop_size, only_horizontal, res):
        output = extend_ndarray_for_prediction(test_input_arr, crop_size=crop_size, only_horizontal=only_horizontal)
        assert (output == res).all()

    
    @pytest.mark.parametrize(
    'input_arr, crop_size, only_horizontal, res',
    [
        (np.array([[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]), 2, False, np.array([[[6],[4],[5],[6],[4]],
                                                        [[3],[1],[2],[3],[1]],
                                                        [[6],[4],[5],[6],[4]],
                                                        [[9],[7],[8],[9],[7]],
                                                        [[6],[4],[5],[6],[4]]])),
        (np.array([[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]), 2, True, np.array([[[3],[1],[2],[3],[1]],
                                                       [[6],[4],[5],[6],[4]],
                                                       [[9],[7],[8],[9],[7]]]))
    ]
    )
    def test_correct_input_but_input_array_has_empty_dims(self, input_arr, crop_size, only_horizontal, res):
        assert (extend_ndarray_for_prediction(input_arr, crop_size, only_horizontal) == res).all()

    
    @pytest.mark.parametrize(
    'input_arr, crop_size',
    [
        # input_arr has less than 2 non empty dims
        (np.array([[1,2,3]]), 1),
        # input_arr first dim is empty 
        (np.array([[[1,2,3],
                    [1,2,3]]]), 1),
        # input_arr second dim is empty 
        (np.array([[[1,2,3]],
                   [[1,2,3]]]), 1),
    ]
    )
    def test_uncorrect_input_when_input_array_has_empty_dims(self, input_arr, crop_size):
        with pytest.raises(ValueError):
            extend_ndarray_for_prediction(input_arr, crop_size)

    
    @pytest.mark.parametrize(
    'crop_size',
    [
        # crop_size bigger than rows quantity 
        (5),
        # crop_size bigger than cols quantity
        (4),
        # negative crop_size
        (-2),
        # zero crop_size
        (0)
    ]
    )
    def test_uncorrect_crop_size_and_crop_step(self, test_input_arr, crop_size):
        with pytest.raises(ValueError):
            extend_ndarray_for_prediction(test_input_arr, crop_size=crop_size)

    
    @pytest.mark.parametrize(
    'input_arr, crop_size',
    [
        # df is not pandas.DataFrame
        ('sdfsdf', 2,),
        # crop_size is not int
        (np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9],
                   [10,11,12]]), 'gfg2')
    ]
    )
    def test_uncorrect_input_values_type(self, input_arr, crop_size):
        with pytest.raises(ValidationError):
            extend_ndarray_for_prediction(input_arr, crop_size=crop_size)


if __name__ == "__main__":
    pytest.main()