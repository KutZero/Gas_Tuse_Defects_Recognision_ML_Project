from custom_modules.data_worker._draw_defects_map import (
    draw_zeros_quantity_in_data_df, draw_defects_map,
    draw_defects_map_with_reference_owerlap,
    draw_defects_map_with_rectangles_owerlap,
    _build_defects_map, _approx_df, _polygonize,
    _add_index_fillers, _cals_labels_paddings, 
    _calc_labels_and_locs)

import pytest
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from contextlib import nullcontext as does_not_raise


class Test_draw_zeros_quantity_in_data_df:
    pass


class Test_draw_defects_map:
    pass


class Test_draw_defects_map_with_reference_owerlap:
    pass


class Test_draw_defects_map_with_rectangles_owerlap:
    pass


class Test__build_defects_map:
    pass


class Test__approx_df:
    pass


class Test__polygonize:
    pass


class Test__add_index_fillers:
    pass


class Test__cals_labels_paddings:
    pass


class Test__calc_labels_and_locs:
    pass


if __name__ == "__main__":
    pytest.main()