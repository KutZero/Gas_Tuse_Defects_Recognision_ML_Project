from custom_modules.data_worker import get_x_and_y_data, \
    pandas_crop_to_image_like_numpy, reshape_df_for_future_crops, \
    reshape_x_df_to_image_like_numpy, reshape_y_df_to_image_like_numpy, \
    normalize_data, standartize_data, split_def_and_non_def_data, \
    create_binary_arr_from_mask_arr, augment_data, \
    split_data_to_train_val_datasets, _calculate_crops_with_defects_positions, \
    _get_df_from_defects_file, _split_cell_string_value_to_numpy_array_of_64_values, \
    _get_df_from_data_file, _is_path_correct

import pytest
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from contextlib import nullcontext as does_not_raise

class Test_get_x_and_y_data:
    pass

class Test_pandas_crop_to_image_like_numpy:
    pass

class Test_reshape_df_for_future_crops:
    pass

class Test_reshape_x_df_to_image_like_numpy:
    pass

class Test_reshape_y_df_to_image_like_numpy:
    pass

class Test_normalize_data:
    pass

class Test_standartize_data:
    pass

class Test_split_def_and_non_def_data:
    pass

class Test_create_binary_arr_from_mask_arr:
    pass

class Test_augment_data:
    pass
    
class Test_split_data_to_train_val_datasets:
    pass
    
class Test__calculate_crops_with_defects_positions:
    pass

class Test__get_df_from_defects_file:
    pass
    
class Test__split_cell_string_value_to_numpy_array_of_64_values:
    pass

class Test__get_df_from_data_file:
    pass

@pytest.fixture
def csv_file_path(tmpdir_factory):
    df = pd.DataFrame({'col1': [1, 2, 1], 
                       'col2': [3, 4, 1],
                       'col2': [5, 6, 1]}) 
    filename = str(tmpdir_factory.mktemp('data').join('file.csv'))
    df.to_csv(filename)
    return filename

# test _is_path_correct func
class Test__is_path_correct:
    @pytest.mark.parametrize(
        'path, res, expectation',
        [
            (2, False, pytest.raises(TypeError)),
            ('data/data', False, pytest.raises(ValueError)),
            ('datrhtrhrtht', False, pytest.raises(ValueError)),
            ('data/data/file.csv', False, pytest.raises(ValueError)),
        ]
    )
    def test_is_path_correct_without_file(self, path, res, expectation):
        with expectation:
            assert _is_path_correct(path) == res

    def test_is_path_correct_with_file(self, csv_file_path):
        assert _is_path_correct(csv_file_path) == True 

if __name__ == "__main__":
    pytest.main()