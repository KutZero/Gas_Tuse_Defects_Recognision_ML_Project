from custom_modules.data_worker._ndarray_utils import (
    extend_ndarray_for_crops_dividing,
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

class Test_extend_ndarray_for_crops_dividing:
    @pytest.mark.parametrize(
    'crop_size, crop_step, res',
    [
        # nothing
        (3, 1, np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])),
        # nothing
        ('3', 1, np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])),
        # nothing
        (3, '1', np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])),
        # nothing
        ('3', '1', np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])),
        # add rows only
        (3, 2, np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[7,8,9]])),
        # add cols only
        (2, 2, np.array([[1,2,3,2], [4,5,6,5],[7,8,9,8],[10,11,12,11]])),
        # add rows and cols
        (2, 3, np.array([[1,2,3,2,1],[4,5,6,5,4],[7,8,9,8,7],[10,11,12,11,10], [7,8,9,8,7]])),
    ]
    )
    def test_correct_input_extend_principle(self, test_input_arr, crop_size, crop_step, res):
        output = extend_ndarray_for_crops_dividing(test_input_arr, crop_size=crop_size, crop_step=crop_step)
        assert (output == res).all()

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
    def test_uncorrect_crop_size_and_crop_step(self, test_input_arr, crop_size, crop_step, expectation):
        with expectation:
            assert extend_ndarray_for_crops_dividing(test_input_arr, crop_size=crop_size, crop_step=crop_step)

    @pytest.mark.parametrize(
    'input_arr, crop_size, crop_step, expectation',
    [
        # array is not numpy.ndarray
        ('sdfsdf', 5, 1, pytest.raises(ValidationError)),
        # crop_size is not int
        (np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), '2grg', 1, pytest.raises(ValidationError)),
        # crop_step in not int
        (np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), 2, 'reger', pytest.raises(ValidationError))
    ]
    )
    def test_uncorrect_input_values_type(self, input_arr, crop_size, crop_step, expectation):
        with expectation:
            assert extend_ndarray_for_crops_dividing(input_arr, crop_size=crop_size, crop_step=crop_step)

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
    def test_uncorrect_crop_size_and_crop_step(self, test_input_arr, crop_size, expectation):
        with expectation:
            assert extend_ndarray_for_prediction(test_input_arr, crop_size=crop_size)

    @pytest.mark.parametrize(
    'input_arr, crop_size, expectation',
    [
        # df is not pandas.DataFrame
        ('sdfsdf', 2, pytest.raises(ValidationError)),
        # crop_size is not int
        (np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), 'gfg2', pytest.raises(ValidationError))
    ]
    )
    def test_uncorrect_input_values_type(self, input_arr, crop_size, expectation):
        with expectation:
            assert extend_ndarray_for_prediction(input_arr, crop_size=crop_size)


if __name__ == "__main__":
    pytest.main()