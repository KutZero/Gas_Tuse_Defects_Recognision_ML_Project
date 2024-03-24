"""
Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import logging
import pandas as pd
import numpy as np

from typing_extensions import Annotated
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field

PositiveInt = Annotated[int, Field(gt=0), AfterValidator(lambda x: int(x))]

# create logger
logger = logging.getLogger('main.'+__name__)

@validate_call(config=dict(arbitrary_types_allowed=True))
def extend_ndarray_for_crops_dividing(arr: np.ndarray, crop_size: PositiveInt, crop_step: PositiveInt) -> np.ndarray:
    """
    Extend the np.ndarray for exact crops dividing by a determined crop window with
    a determined cropping step.

    Parameters
    ----------
    arr : np.ndarray
        The np.ndarray to be extended.
    crop_size : int
        The size of crop sliding window (has equal sides).
    crop_step : int
        The step for df cropping by sliding window).

    Returns
    -------
    out : np.ndarray
        Output extended np.ndarray.
        
    """ 
    message = f"""
    The input ndarray shape: {arr.shape}"""
    new_rows = crop_step - ((arr.shape[0] - crop_size) % crop_step)
    new_cols = crop_step - ((arr.shape[1] - crop_size) % crop_step)

    if new_rows != crop_step:
        arr = np.pad(arr, ((0, new_rows),*[(0,0) for i in range(arr.ndim-1)]), 'reflect')
    if new_cols != crop_step:
        arr = np.pad(arr, ((0, 0),(0, new_cols),*[(0,0) for i in range(arr.ndim-2)]), 'reflect')

    logger.debug(f"""{message}
    The crop size: {crop_size}
    The crop step: {crop_step}
    The output ndarray shape: {arr.shape}""")
    
    return arr

@validate_call(config=dict(arbitrary_types_allowed=True))
def extend_ndarray_for_prediction(arr: np.ndarray, crop_size: PositiveInt) -> np.ndarray:
    """
    Extend np.ndarray for increasing network model prediction or
    training quantity. 
    To left side of dfs is added reverse right crop_size-1 
    columns of them. 
    To right side of dfs is added reverse left crop_size-1 
    columns of them and some for exact dividing by crops with
    crop_size and crop_step. 
    To top side is added reverse top crop_size-1 rows. 
    To bottom side is added reverse bottom crop_size-1 rows
    and some for exact dividing by crops with crop_size and 
    crop_step.

    Parameters
    ----------
    arr : np.ndarray
        The np.ndarray to be extended.
    crop_size : int
        The size of crop sliding window (has equal sides).
    crop_step : int
        The step for df cropping by sliding window).

    Returns
    -------
    out : np.ndarray
        Output extended np.ndarray.
        
    """
    message = f"""
    The input ndarray shape: {arr.shape}"""
    arr = np.pad(arr, ((crop_size-1, crop_size-1),(0, 0),(0, 0)), 'reflect')
    arr = np.pad(arr, ((0, 0),(crop_size-1, crop_size-1),(0, 0)), 'wrap')
    
    logger.debug(f"""{message}
    The crop size: {crop_size}
    The output ndarray shape: {arr.shape}""")
    
    return arr