__all__ = ['get_crop_generator', 'get_x_and_y_data_dfs']

import logging
import itertools
import pandas as pd
import numpy as np
import os
import re
from pydantic import validate_call, PositiveInt
from typing import Callable, Optional, Generator, Iterable
import pathlib

from custom_modules.data_worker import DataPart

# create logger
logger = logging.getLogger('main.'+__name__)


@validate_call(config=dict(arbitrary_types_allowed=True))
def get_crop_generator(arr: np.ndarray, 
                       crop_size: PositiveInt, 
                       crop_step: PositiveInt, 
                       augmentations: bool = False) -> Iterable[np.array]:
    """
    Create generator for sliding window across arr with given step and crop size

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array with crops data
    crop_size: PositiveInt
        The size of the square crop side
    crop_step: PositiveInt
        The size of the sliding window
        step making crops
    augmentations: bool
        Apply of not augmentations to the arr
        sliding crops. Augmentations to be used: 
        rotation for 90 degree (4 times); 
        horizontal and vertical mirroring.

    Yields
    -------
    out : numpy.ndarray
        The numpy array with sliding crops data
    """
    in_arr_shape = arr.shape
        
    logger.debug(f'''
    The input array cells: {in_arr_shape} 
    The crop size: {crop_size} 
    The crop step: {crop_step}''')

    arr = dw.match_ndarray_for_crops_dividing(arr, crop_size, crop_step, mode='crop')

    if arr.shape != in_arr_shape:
        logger.warning(f'''
    The arr with shape = {in_arr_shape} cant be divided by {crop_size=} with {crop_step=}. 
    The custom_modules.data_worker.match_ndarray_for_crops_dividing() with mode="crop" used. The new
    arr shape={arr.shape}''')
    
    if augmentations:
        arr1 = np.concatenate([arr, np.flip(arr,0)],axis=0)
        arr1 = np.concatenate([arr1, np.flip(arr1,1)],axis=0)
        arr2 = np.rot90(arr1,1,[0,1])
        arrs = np.split(arr1, 4, axis=0) + np.split(arr2, 4, axis=1)

        for arr in arrs:
            for i in range(0, arr.shape[0] - crop_size + 1, crop_step):  
                for j in range(0, arr.shape[1] - crop_size + 1, crop_step):  
                    yield arr[i:i+crop_size, j:j+crop_size]

    else:
        for i in range(0, arr.shape[0] - crop_size + 1, crop_step):  
            for j in range(0, arr.shape[1] - crop_size + 1, crop_step):  
                yield arr[i:i+crop_size, j:j+crop_size]


@validate_call(config=dict(arbitrary_types_allowed=True))
def get_x_and_y_data_dfs(data_part: dw.DataPart) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data from set of .csv files

    Parameters
    ----------
    data_part_desc : DataPart
        The describtion of data to read. For more info 
        see DataPart docs

    Returns
    -------
    data_df : pandas.DataFrame
        The dataframe with data got from detectors
    defects_df : pandas.DataFrame
        The dataframe with data got from specialists
        about defects depths and locations for data_df
        
    """
    logger.debug(f"""
    The input data_part: {data_part}""")

    data_df, defects_df = _read_data_df_and_defects_df(data_part.data_path, 
                                                       data_part.defects_path)

    data_df, defects_df = _unify_dfs(data_df, 
                                     defects_df, 
                                     data_part.unify_func, 
                                     data_part.unify_separatly)

    data_df, defects_df = _crop_data_df_and_defects_df(data_df, 
                                                       defects_df, 
                                                       data_part.xy, 
                                                       data_part.width, 
                                                       data_part.height)

    return data_df, defects_df


@validate_call(config=dict(arbitrary_types_allowed=True))
def _read_data_df_and_defects_df(data_path: os.PathLike, 
                                 defects_path: os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_df = _get_df_from_data_file(data_path)
    defects_df = _get_df_from_defects_file(defects_path)
    
    # create defects depths mask
    # create base zeros dataframe with size like data_df
    base_df = pd.DataFrame(data = 0.0, index = data_df.index,
                           columns = data_df.columns)
    # read line-by-line defects_df
    # get defects location and mark by ones
    for row_name in defects_df.index.values.tolist():
        (row_min, row_max,
         detector_min, detector_max,
         fea_depth ) = defects_df.astype('object').loc[row_name].to_list()
        
        # mark defect location in base dataframe
        if (detector_min < detector_max):
            base_df.iloc[row_min:row_max+1,detector_min:detector_max+1] = fea_depth
            continue
    
        base_df.iloc[row_min:row_max+1,detector_min:data_df.shape[1]] = fea_depth
        base_df.iloc[row_min:row_max+1,:detector_max+1] = fea_depth
    defects_df = base_df

    logger.debug(f"""
    Read detectors data shape: {data_df.shape}
    Read defect data shape: {defects_df.shape}""")
    return data_df, defects_df


@validate_call(config=dict(arbitrary_types_allowed=True))
def _unify_dfs(data_df: pd.DataFrame, 
               defects_df: pd.DataFrame, 
               unify_func: Optional[Callable[[np.array],np.array]] = None, 
               unify_separatly: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if unify_func is not None:
        data_df_values = dw.df_to_numpy(data_df)
        defects_df_values = defects_df.to_numpy()
        if unify_separatly:
            logger.debug(f"""\nUnifying data separatly with {unify_func.__name__} func...""")
            x_arr = np.concatenate([unify_func(data_df_values[:,:,:32]), unify_func(data_df_values[:,:,32:])],axis=2)
        else:
            logger.debug(f"""\nUnifying data together with {unify_func.__name__} func...""")
            x_arr = unify_func(data_df_values)
        y_arr = unify_func(defects_df_values)

        data_df = pd.DataFrame(data=0, index=data_df.index, columns=data_df.columns, dtype='object')
        for i in range(data_df.shape[0]):
            for j in range(data_df.shape[0]):
                data_df.iloc[i,j] = data_df_values[i,j]
        defects_df = pd.DataFrame(data=defects_df_values, index=defects_df.index, columns=defects_df.columns)
    return data_df, defects_df
    

@validate_call(config=dict(arbitrary_types_allowed=True))
def _crop_data_df_and_defects_df(data_df: pd.DataFrame, 
                                 defects_df: pd.DataFrame, 
                                 xy: tuple[int,int] = (0,0),
                                 width: Optional[PositiveInt] = None, 
                                 height: Optional[PositiveInt] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if xy == (0,0) and height is None and width is None:
        return data_df, defects_df

    end_row = None if height is None else xy[1]+height
    end_col = None if width is None else xy[0]+width
    
    data_df = data_df.iloc[xy[1]:end_row, xy[0]:end_col]
    defects_df = defects_df.iloc[xy[1]:end_row, xy[0]:end_col]

    logger.debug(f"""
    Cropped with (xy={xy},width={width},height={height}) detectors data shape: {data_df.shape}
    Cropped with (xy={xy},width={width},height={height}) defect data shape: {defects_df.shape}""")
    
    return data_df, defects_df


@validate_call(config=dict(arbitrary_types_allowed=True))
def _get_df_from_defects_file(path_to_defects_file: os.PathLike) -> pd.DataFrame:
    """Read data file like "*_defects.csv" and returns
       pandas dataframe with preprocessed data from it"""
    using_columns = ['row_min', 'row_max', 'detector_min', 
                     'detector_max', 'fea_depth']
    df = pd.read_csv(path_to_defects_file, delimiter=';')
    return df[using_columns]


@validate_call(config=dict(arbitrary_types_allowed=True))
def _split_cell_string_value_to_numpy_array_of_64_values(df_cell_value: str) -> np.ndarray:
    """Converte all data cells values from given pandas dataframe from
    string (describes 2D values array) to 1D float numpy array of 64 items"""
    num_pars = re.findall(r'(-?\d+(\.\d+)*)*\s*:\s*(-?\d+(\.\d+)*)*', df_cell_value)

    if len([item[0] for item in num_pars if item[0] and item[2]]) == 0:
        #logger.debug(f"""Got cell value without any full time-value pars""")
        return np.zeros((64))
    
    for item in num_pars:
        if not item[0] or not item[2]:
            logger.debug(f"""Got input df cell str with uncompleted 
            time-amplitude value pars. Uncompleted pars deleted.
            Input str: {df_cell_value}""")
            
    num_pars = np.array([[item[0], item[2]] for item in num_pars if item[0] and item[2]]).astype(float)

    if num_pars.shape[0] > 32:
        raise ValueError(f'Too much time-amplitude values pars in a cell. Got: {num_pars.shape[0]}. Max: 32')
    
    time_vals = num_pars[:,0]
    amp_vals = num_pars[:,1]
    
    time_vals = np.pad(time_vals, (abs(time_vals.size-32), 0), constant_values=(0))
    amp_vals = np.pad(amp_vals, (abs(amp_vals.size-32), 0), constant_values=(0))
    return np.concatenate((time_vals, amp_vals), axis=0)


@validate_call(config=dict(arbitrary_types_allowed=True))
def _get_df_from_data_file(path_to_data_file: os.PathLike) -> str:
    """Read data file like "*_data.csv" and returns
       pandas dataframe with preprocessed data from it"""
    df = pd.read_csv(path_to_data_file, delimiter=';').astype(str)
    df = df.drop(['position'], axis=1)
    df = df.set_index('row')
    df = df.map(_split_cell_string_value_to_numpy_array_of_64_values)
    return df