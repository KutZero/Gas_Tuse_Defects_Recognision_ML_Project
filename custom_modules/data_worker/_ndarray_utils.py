"""
Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import logging
import pandas as pd
import numpy as np

from pydantic import validate_call, PositiveInt
from functools import wraps

# create logger
logger = logging.getLogger('main.'+__name__)


def check_is_arr_valid_decorator(func):
    @wraps(func)
    def wrapper(arr,*args,**kvargs):
        if arr.ndim < 2:
            raise ValueError(f'The input array should have at least 2 ndims, but got {arr.ndim=}')
        if arr.shape[0] == 1:
            raise ValueError(f"The input array first ndim should be at least size of 2, but got {arr.shape=}")
        if arr.shape[1] == 1:
            raise ValueError(f"The input array's second ndim should be at least size of 2, but got {arr.shape=}")  
        return func(arr,*args,**kvargs)
    return wrapper

    
@validate_call(config=dict(arbitrary_types_allowed=True))
@check_is_arr_valid_decorator
def match_ndarray_for_crops_dividing(arr: np.ndarray, 
                                     crop_size: PositiveInt, 
                                     crop_step: PositiveInt,
                                     mode: str = 'extend') -> np.ndarray:
    """
    Match the np.ndarray for exact crops dividing by a determined crop window with
    a determined cropping step via extending or cropping it.

    Parameters
    ----------
    arr : np.ndarray
        The np.ndarray to be extended.
    crop_size : int
        The size of crop sliding window (has equal sides).
    crop_step : int
        The step for df cropping by sliding window).
    mode: str
        The mode to use when preparing the numpy array for
        crop dividing. Can be "extend" or "crop".

    Returns
    -------
    out : np.ndarray
        Output match np.ndarray.
        
    """ 
    message = f"""
    The input ndarray shape: {arr.shape}"""
    
    arr = arr.copy()
    if not mode in ('crop','extend'):
        raise ValueError(f'The mode param should be one of the folowing: crop; extend. Got {mode=}') 
    if crop_size > arr.shape[0] or crop_size > arr.shape[1]:
        raise ValueError("Crop size should be bigger or equal " + 
                         f"than the arr less axis. The {arr.shape=}, but got the {crop_size=}")
    if crop_step > arr.shape[0] or crop_step > arr.shape[1]:
        raise ValueError("Crop step should be bigger or equal " + 
                         f"than the arr less axis. The {arr.shape=}, but got the {crop_step=}")
    
    new_rows = crop_step - ((arr.shape[0] - crop_size) % crop_step)
    new_cols = crop_step - ((arr.shape[1] - crop_size) % crop_step)

    if new_rows != crop_step:
        if mode == 'extend':
            arr = np.pad(arr, ((0, new_rows),*[(0,0) for i in range(arr.ndim-1)]), 'reflect')
        elif mode == 'crop':
            arr = arr[:-1*(crop_step-new_rows)]
    if new_cols != crop_step:
        if mode == 'extend':
            arr = np.pad(arr, ((0, 0),(0, new_cols),*[(0,0) for i in range(arr.ndim-2)]), 'reflect')
        elif mode == 'crop':
            arr = arr[:,:-1*(crop_step-new_cols)]
            

    logger.debug(f"""{message}
    The crop size: {crop_size}
    The crop step: {crop_step}
    The output ndarray shape: {arr.shape}""")
    
    return arr


@validate_call(config=dict(arbitrary_types_allowed=True))
@check_is_arr_valid_decorator
def extend_ndarray_for_prediction(arr: np.ndarray, crop_size: PositiveInt, only_horizontal: bool=False) -> np.ndarray:
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
    only_horizontal: bool
        If true extends the df only horizontally
        
    Returns
    -------
    out : np.ndarray
        Output extended np.ndarray.
        
    """
    message = f"""
    The input ndarray shape: {arr.shape}"""      

    arr = arr.copy()
    if crop_size > arr.shape[0] or crop_size > arr.shape[1]:
        raise ValueError("Crop size should be bigger or equal " + 
                         f"than the arr less axis. The {arr.shape=}, but got the {crop_size=}")
    
    if not only_horizontal:
        arr = np.pad(arr, ((crop_size-1, crop_size-1),*[(0,0) for i in range(arr.ndim-1)]), 'reflect')
    arr = np.pad(arr, ((0, 0),(crop_size-1, crop_size-1),*[(0,0) for i in range(arr.ndim-2)]), 'wrap')
    
    logger.debug(f"""{message}
    The crop size: {crop_size}
    The output ndarray shape: {arr.shape}""")
    
    return arr