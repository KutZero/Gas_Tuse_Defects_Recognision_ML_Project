from custom_modules.data_worker.data_worker import (
    calc_model_prediction_accuracy,
    normalize_data,
    standardize_data,
    )

import pytest
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import itertools

from contextlib import nullcontext as does_not_raise
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field


class Test_calc_model_prediction_accuracy:
    pass

class Test_normalize_data:
    def test_zeros_input_array(self):
        input_value = np.zeros((2,2), dtype='float32')
        assert (normalize_data(input_value) == input_value).all()

    @pytest.mark.parametrize(
    'input_value, res',
    [
        # positive values array
        (np.array([1.,2.,3.,4.]), np.array([0.25, 0.5, 0.75, 1])),
        # positive and negative values array
        (np.array([1.,-2.,3.,-5.]), np.array([0.75, 0.375, 1., 0.])),
    ])
    def test_correct_input(self, input_value, res):
        assert (normalize_data(input_value) == res).all()

class Test_standardize_data:
    def test_zeros_input_array(self):
        input_value = np.zeros((2,2), dtype='float32')
        assert (standardize_data(input_value) == input_value).all()

    @pytest.mark.parametrize(
    'input_value, res',
    [
        # positive values array
        (np.array([1.,2.,3.,4.]), np.array([0.25, 0.5, 0.75, 1])),
        # positive and negative values array
        (np.array([1.,-2.,3.,-4.]), np.array([0.25, -0.5, 0.75, -1])),
    ])
    def test_correct_input(self, input_value, res):
        assert (standardize_data(input_value) == res).all()

if __name__ == "__main__":
    pytest.main()