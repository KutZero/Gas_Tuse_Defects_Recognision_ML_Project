from custom_modules.pipe_data._path_to_csv import PathToCsv

import pytest
import pandas as pd

from contextlib import nullcontext as does_not_raise

@pytest.fixture
def csv_file_path(tmpdir_factory):
    df = pd.DataFrame({'col1': [1, 2], 
                       'col2': [3, 4]}) 
    filename = str(tmpdir_factory.mktemp('data').join('file.csv'))
    df.to_csv(filename)
    return filename

class TestPathToCsv:
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
            assert PathToCsv().is_path_correct(path) == res

    def test_is_path_correct_with_file(self, csv_file_path):
        assert PathToCsv().is_path_correct(csv_file_path) == True 

if __name__ == "__main__":
    pytest.main()